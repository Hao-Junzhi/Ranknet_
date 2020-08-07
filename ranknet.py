import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataloader import L2RDataset
import numpy as np
import math


def design_model(actf, dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(actf)
    layers.append(nn.Linear(dims[-1], 1))
    return layers
# 在dim预设好每一层的神经元个数，比如说dims = [46, 128, d=（32，64，28）, 32]。
# d可以设置为32，64和128中的一个。分别完成结果，结果选表现最好的那个神经网络。
# 该函数作用为设计神经网络的每一层。然后在构建一个具体的RankNet类。


# 定义一个类作为RankNet训练模型
class RankNet(nn.Module):
    def __init__(self, layers):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(*layers) #构建神经网络

    def forward(self, batch_ranking=None, batch_stds_labels=None, sigma=1.0):
        s_batch = self.model(batch_ranking)
        #这里batch_ranking指代一个特定查询词下，第n个数据的那组的46维的向量
        # s_batch指n个数据的预测得分score
        pred_diff = s_batch - s_batch.view(1, s_batch.size(0))
        # 将文档对相关数据变为一张矩阵表，实际上只取上半张即可。每一项代表一对文档的预测的相关度关系。
        row_inds, col_inds = np.triu_indices(batch_ranking.size()[0], k=1)
        si_sj = pred_diff[row_inds, col_inds]
        # 预测相关性值之差
        std_diffs = batch_stds_labels.view(batch_stds_labels.size(0), 1) - batch_stds_labels.view(1,batch_stds_labels.size(0))
        Sij = torch.clamp(std_diffs, min=-1, max=1)
        Sij = Sij[row_inds, col_inds]
        # 真实相关性之差
        batch_loss_1st = 0.5 * sigma * si_sj * (1.0 - Sij)
        batch_loss_2nd = torch.log(torch.exp(-sigma * si_sj) + 1.0)
        batch_loss = torch.sum(batch_loss_1st + batch_loss_2nd)
        # 带入 RankNet计算公式，计算loss值，并将值返回 公式为C_ij=1/2*(1-S_ij)*σ(s_i-s_j)+log[1+e^-σ(s_i-s_j)] 来自交叉熵函数的推导
        return batch_loss

    def predict(self, x):
        return self.model(x)

def train_step(model, l2r_dataset, optimizer):
    epoch_loss_ls = []
    for batch_rankings, batch_std_labels in l2r_dataset:
        loss = model(batch_ranking=batch_rankings, batch_stds_labels=batch_std_labels, sigma=1.0)
        epoch_loss_ls.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(epoch_loss_ls) / len(epoch_loss_ls)
# 训练，梯度清零，反向传播，更新参数空间。函数返回每个epoch的平均loss

def test_step(model, test_ds):
    results = {}
    for k in [1, 3, 5, 10]:
        ndcg_ls = []
        # nDCG@1，@3，@5，@10
        for batch_rankings, labels in test_ds:
            pred = model.predict(batch_rankings)
            # pred 用现有训练模型的预测值
            pred_ar = pred.squeeze(1).detach()
            # 去掉所有维度为1的维度
            label_ar = labels.detach()
            _, argsort = torch.sort(pred_ar, descending=True, dim=0)
            # 按行从大到小排
            pred_ar_sorted = label_ar[argsort]
            # 预测数据与实际对上号
            if len(pred_ar_sorted) >= k:
                ndgc_s = ndcg_score(label_ar, pred_ar_sorted, k=k)
                if not math.isnan(ndgc_s):
                    ndcg_ls.append(ndgc_s)
            # 检索词下的相关结果不够ndcg@的情况
        results[k] = sum(ndcg_ls) / len(ndcg_ls)

    return results


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    def dcg_score(y_score, k=k, gains="exponential"):
        y_score_k = y_score[:k]

        if gains == "exponential":
            gains = torch.pow(2.0, y_score_k) - 1.0
            gains = gains.type(torch.FloatTensor)
        elif gains == "linear":
            gains = y_score[:k]
        else:
            raise ValueError("Invalid gains option.")
        # 因为MQ2007相关性标签只有1或0 所以二者实际上等价
        discounts = torch.log2(torch.arange(k).type(torch.FloatTensor) + 2)
        # 因为从0开始所以+2
        return torch.sum(gains / discounts)

    best = dcg_score(y_true, k, gains)
    actual = dcg_score(y_score, k, gains)
    result = actual / best
    # ndcg 就是用现有除以最优（实际）
    return result.item()


if __name__ == '__main__':

    for d in [32, 64, 128]:
        now = datetime.datetime.now()
        now = "{0:%Y%m%d%H%M}".format(now)
        w = SummaryWriter()
        file = open('./result.txt'.format(d, now), 'w')
        total_ndcg = {}
        for k in [1, 3, 5, 10]:
            total_ndcg[k] = 0
        total_loss = 0
        dims = [46, 128, d, 32]
        # 神经网络定义层数
        models = {}
        actf1 = nn.ReLU()
        actf2 = nn.Sigmoid()
        # 定义激活函数
        max_epoch = 50
        file.write("epoch: {}".format(max_epoch))
        file.write('\n')

        for n in ['1', '2', '3', '4', '5']:
            layers = design_model(actf1, dims)
            models[n] = RankNet(layers)
            # 用5个交叉数据集，分别制造模型并验证
            optimizer = torch.optim.Adam(models[n].parameters(), lr=0.001)
            train_file = 'C:/Users/Junzhi Hao/Desktop/MQ2007/Fold%s/train.txt' % n
            val_file = 'C:/Users/Junzhi Hao/Desktop/MQ2007/Fold%s/vali.txt' % n
            test_file = 'C:/Users/Junzhi Hao/Desktop/MQ2007/Fold%s/test.txt' % n
            train_ds = L2RDataset(file=train_file, data_id='MQ2007_Super')
            val_ds = L2RDataset(file=val_file, data_id='MQ2007_Super')
            test_ds = L2RDataset(file=test_file, data_id='MQ2007_Super')
            best_val_ndcg_score = 0
            if n == '1':
                model_s = str(models[n])
                file.write(model_s)

            for epoch in range(max_epoch):
                epoch_train_loss = train_step(models[n], train_ds, optimizer)
                print("Epoch: {} Train Loss: {}".format(epoch, epoch_train_loss))
                # 训练一次
                epoch_train_dcg = test_step(models[n], train_ds)
                for k in [1, 3, 5, 10]:
                    print("Epoch: {} Train nDCG@{}: {}".format(epoch, k, epoch_train_dcg[k]))
                    w.add_scalar("train nDCG@%d %s" % (k, n), epoch_train_dcg[k], epoch)
                # 计算训练集的ndcg@1，3，5，10，并写入记事本
                w.add_scalar("train loss %s" % n, epoch_train_loss, epoch)
                epoch_val_dcg = test_step(models[n], val_ds)
                # 上边计算train。下边计算vali
                for k in [1, 3, 5, 10]:
                    print("Epoch: {} Val nDCG@{}: {}".format(epoch, k, epoch_val_dcg[k]))
                    w.add_scalar("val nDCG@%d %s" % (k, n), epoch_val_dcg[k], epoch)
                if epoch_val_dcg[10] > best_val_ndcg_score:
                    best_epoch = epoch
                    best_loss = epoch_train_loss
                    best_val_ndcg_score = epoch_val_dcg[10]
                    torch.save(models[n], 'C://Users//Junzhi Hao//Desktop//MQ2007//Fold%s//model' % n)
                print("--" * 50)

            val_model = torch.load('C://Users//Junzhi Hao//Desktop//MQ2007//Fold%s//model' % n)
            test_ndcg = test_step(val_model, test_ds)
            for k in [1, 3, 5, 10]:
                total_ndcg[k] += test_ndcg[k]
                print("--" * 50)
                print("Test NDCG@{}: {}".format(k, test_ndcg[k]))
                print("--" * 50)
                file.write("Folder: {} Test NDCG@{}: {}".format(n, k, test_ndcg[k]))
                file.write('\n')
            # 验证集的nDCG@
            file.write("Best epoch : {}".format(best_epoch))
            file.write('\n')
            file.write("Best train loss : {}".format(best_loss))
            file.write('\n')
            total_loss += best_loss
        for k in [1, 3, 5, 10]:
            ave_ndcg = total_ndcg[k] / 5
            print("Ave Test NDCG@{}: {}".format(k, ave_ndcg))
            file.write("Ave Test NDCG@{}: {}".format(k, ave_ndcg))
            file.write('\n')
        ave_loss = total_loss / 5
        file.write("Ave train loss in best : {}".format(ave_loss))
        # 计算5折交叉验证的平均loss
        w.close()
