import os
import torch
import pickle
import pandas as pd
import numpy as np
import torch.utils.data as data
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# 最大字节不超过4GB
max_bytes = 2 ** 31 - 1
# 归一化，根据鲁棒性缩放数据，标准化特征
SCALER_LEVEL = ['QUERY', 'DATASET']
SCALER_ID = ['MinMaxScaler', 'RobustScaler', 'StandardScaler']
# Supervised ranking，Semi-supervised ranking，Listwise ranking
MSLETOR = ['MQ2007_Super', 'MQ2008_Super', 'MQ2007_Semi', 'MQ2008_Semi', 'MQ2007_List', 'MQ2008_List']
MSLETOR_SUPER = ['MQ2007_Super', 'MQ2008_Super']
MSLETOR_SEMI = ['MQ2007_Semi', 'MQ2008_Semi']
MSLETOR_LIST = ['MQ2007_List', 'MQ2008_List']
MSLRWEB = ['MSLRWEB10K', 'MSLRWEB30K']
# 封存 pickle
def pickle_save(target, file):
    bytes_out = pickle.dumps(target, protocol=4) #将目标封存，默认协议版本是4
    with open(file, 'wb') as f_out:              #读，二进制
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes]) #将数据全部写入

# 同上，逆过来读取
def pickle_load(file):
    file_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        bytes_in = bytearray(0)
        for _ in range(0, file_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data


class L2RDataLoader():
    """
	learning-to-rank dataloader
	"""
    def __init__(self, file, buffer=True):
        '''
		:param file:  k折的源目录
		:param buffer: 是否要缓冲区，预加载
		'''
        self.df = None
        self.file = file
        self.buffer = buffer

    def load_data(self):
        pass

    def filter(self):
        pass


class MSL2RDataLoader(L2RDataLoader):
    """
	The data loader for MS learning-to-rank datasets
	"""

    def __init__(self, file, data_id=None, buffer=True):
        super(MSL2RDataLoader, self).__init__(file=file, buffer=buffer)

        self.data_id = data_id
        # origianl data as dataframe
        self.df_file = file[:file.find('.txt')].replace('Fold','BufferedFold') + '.df'
        # 以bufferedFold作为缓冲进行训练，后缀名改为.df

        pq_suffix = 'PerQ'

        # plus scaling
        self.scale_data = True
        self.scaler_id = 'StandardScaler'

        if self.scale_data:
            pq_suffix = '_'.join([pq_suffix, 'QS', self.scaler_id])

        self.perquery_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '_' + pq_suffix + '.np'
        # 将检索关键词后缀改为.np

    def load_data(self):
        '''
		Load data at a per-query unit consisting of {scaled} {des-sorted} document vectors and standard labels
		:param given_scaler: scaler learned over entire training data, which is only needed for dataset-level scaling
		:return:
		'''
        if self.data_id in MSLETOR:
            self.num_features = 46
        elif self.data_id in MSLRWEB:
            self.num_features = 136
            # 略

        self.feature_cols = [str(f_index) for f_index in range(1, self.num_features + 1)]
        # 向量标签

        if os.path.exists(self.perquery_file):
            list_Qs = pickle_load(self.perquery_file)
            return list_Qs
        else:
            self.get_df_file()
            # df= 整理后读取的数据

            self.ini_scaler()
            # 数据清洗方式

            list_Qs = []
            qids = self.df.qid.unique()
            np.random.shuffle(qids)
            # 提取所有不重复的qid，并随机
            for qid in qids:
                sorted_qdf = self.df[self.df.qid == qid].sort_values('rele_truth', ascending=False)
                # 按照query来存放数据
                doc_reprs = sorted_qdf[self.feature_cols].values
                # 将特定的query的数据整合为一个向量 doc_reqrs
                if self.scale_data:
                    doc_reprs = self.scaler.fit_transform(doc_reprs)
                    # normalization MinMaxScaler()
                doc_labels = sorted_qdf['rele_truth'].values
                # doc_ids    = sorted_qdf['#docid'].values
                # commented due to rare usage

                list_Qs.append((qid, doc_reprs, doc_labels))

            if self.buffer: pickle_save(list_Qs, file=self.perquery_file)

            return list_Qs
            #
            # ('10', array([[0.004901, 0.      , 0.5     , ..., 0.833333, 0.54023 , 0.      ],
            #  [0.005445, 0.      , 0.5     , ..., 0.333333, 0.252874, 0.      ],
            #  [0.034032, 0.166667, 0.      , ..., 0.333333, 0.321839, 0.      ],
            #  ...,
            #  [0.005717, 0.166667, 0.25    , ..., 0.333333, 0.264368, 0.      ],
            # [0.039477, 0.      , 0.75    , ..., 0.333333, 0.218391, 0.      ],
            # [0.029132, 0.166667, 0.      , ..., 0.5     , 0.37931 , 0.      ]],
            # dtype=float32), array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
            # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0.], dtype=float32))
    def get_df_file(self):
        ''' 加载数据文件，作为dataframe. '''

        if os.path.exists(self.df_file):
            self.df = pd.read_pickle(self.df_file)
        else:
            if self.data_id in MSLETOR:
                self.df = self.load_LETOR4()
            elif self.data_id in MSLRWEB:
                self.df = self.load_MSLRWEB()

            if self.buffer:
                parent_dir = Path(self.df_file).parent
                if not os.path.exists(parent_dir): os.makedirs(parent_dir)
                self.df.to_pickle(self.df_file)

    def load_LETOR4(self):
        '''  '''
        df = pd.read_csv(self.file, sep=" ", header=None)
        # 用panda自带的函数 read_csv读取数据
        df.drop(columns=df.columns[[-2, -3, -5, -6, -8, -9]], axis=1, inplace=True)  # 移除不需要的标签
        # print(self.num_features, len(df.columns) - 5)
        assert self.num_features == len(df.columns) - 5

        for c in range(1, self.num_features + 2):
            # remove keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1]) # 把冒号前面的标签去掉
        df.columns = ['rele_truth', 'qid'] + self.feature_cols + ['#docid', 'inc', 'prob']
        # 相关性，检索词的id，1~46号向量，id，inc，prob

        if self.data_id in MSLETOR_SEMI and self.data_dict['unknown_as_zero']:
            self.df[self.df[self.feature_cols] < 0] = 0
            #待补

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)
            # 转化为浮点 np

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)
        # 增加一列，用于之后过滤数据

        return df

    def load_MSLRWEB(self):
        '''  '''
        df = pd.read_csv(self.file, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True)  # remove the line-break
        assert self.num_features == len(df.columns) - 2

        for c in range(1, len(df.columns)):  # remove the keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

        df.columns = ['rele_truth', 'qid'] + self.feature_cols

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)  # additional binarized column for later filtering

        return df

    def ini_scaler(self):
        assert self.scaler_id in SCALER_ID
        if self.scaler_id == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif self.scaler_id == 'RobustScaler':
            self.scaler = RobustScaler()
        elif self.scaler_id == 'StandardScaler':
            self.scaler = StandardScaler()

# L2RDataset 该类主要让数据类型转化为tensor并提供查找

class L2RDataset(data.Dataset):
    '''
	Buffering tensored objects can save much time.
	'''

    def __init__(self, file, data_id):
        loader = MSL2RDataLoader(file=file, data_id=data_id)
        perquery_file = loader.perquery_file

        torch_perquery_file = perquery_file.replace('.np', '.torch')

        if os.path.exists(torch_perquery_file):
            self.list_torch_Qs = pickle_load(torch_perquery_file)
        else:
            self.list_torch_Qs = []

            list_Qs = loader.load_data()
            list_inds = list(range(len(list_Qs)))
            for ind in list_inds:
                qid, doc_reprs, doc_labels = list_Qs[ind]

                torch_batch_rankings = torch.from_numpy(doc_reprs).type(torch.FloatTensor)

                torch_batch_std_labels = torch.from_numpy(doc_labels).type(torch.FloatTensor)

                self.list_torch_Qs.append((qid, torch_batch_rankings, torch_batch_std_labels))
                # 数据类型转化为tensor

            # buffer
            pickle_save(self.list_torch_Qs, torch_perquery_file)

    def __getitem__(self, index):
        qid, torch_batch_rankings, torch_batch_std_labels = self.list_torch_Qs[index]
        # 选取第n条已经处理好的数据
        return torch_batch_rankings, torch_batch_std_labels

    def __len__(self):
        return len(self.list_torch_Qs)


def transform_ls(q_sample_ls, cols_to_drop):
    """
        input dataframe
        transforms dataframe into tensor
    """

    label_tensor_ls = torch.tensor(np.asarray([q_sample['y'] for q_sample in q_sample_ls]))
    data_tensor_ls = torch.tensor(np.asarray([q_sample[feature_cols].values.astype('float') \
                    for q_sample in q_sample_ls])).float()
    return {'y': label_tensor_ls, 'data': data_tensor_ls}

class RANKNET_TEST_DS(Dataset):
    """Document Ranking Dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the txt file with q_id.
            root_dir (string): Directory with all the query_details.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feats_to_drop = feats_to_drop

    def __len__(self):
        return len(self.meta_file)

    def __getitem__(self, idx):
        q_fname = os.path.join(self.root_dir,str(self.meta_file.iloc[idx]['qid']))
        q_data = pd.read_csv("{}.csv".format(q_fname))
        z_ls = [q_data.iloc[i] for i in range(len(q_data))]
        sample_ls = transform_ls(z_ls,self.feats_to_drop)
        return sample_ls

feats_to_drop = ['doc_id','inc','prob','qid','y']
feature_cols = [str(i) for i in range(1, 47)]