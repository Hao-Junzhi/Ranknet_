#排序学习评价指标解析

##简介

排序学习是利用机器学习方法在数据集上对大量的排序特征进行组合训练, 自动学习参数, 优化评价指标以产生排序模型。

排序学习评价指标通常用于度量排序模型的性能, 如信息检索中常用的准确率(Precesion, P)、召回率(Recall, R)、平均精度均值(Mean average precision, MAP)、归一化折扣累积增益(Normalized discounted cumulated gain, NDCG)、期望倒数排序(Expected reciprocal rank, ERR)等。

## 归一化折扣累积增益(Normalized discounted cumulated gain, NDCG)

NDCG用作排序结果的评价指标，能直接评价排序的准确性。在nDCG中，文档的相关度可以分多个等级进行打分。

推荐系统通常为某用户返回一个item列表，假设列表长度为K，这时可以用NDCG@K评价该排序列表与用户真实交互列表的差距。

NDCG的设立主要遵循着两个思想：　　

1、高关联度的结果比一般关联度的结果更影响最终的指标得分；

2、有高关联度的结果出现在更靠前的位置的时候，指标会越高；



NDCG分别代表Normalized，discounted，cumulated gain。

**Gain（增益）：**表示列表中每一个item的相关性分数，记作$rel_i$，代表i这个位置上这个item的相关度。

**Cumulative Gain（累计增益,CG）：**代表对K个item的Gain进行累加，记作$CG_p=\sum\limits _{i=1}^p rel_i$。只考虑了相关程度的累计增益而没有考虑位置的影响。

**Discounted Cumulated Gain(折损累计增益,DCG）:**在CG的基础上，考虑到了排序顺序的因素，使得排名靠前的item增益更高，对排名靠后的item进行折损。目的就是为了让排名越靠前的结果越能影响最后的结果。这个折损因子为$\frac{1}{log_2(i+1)}$。

故$DCG_p=\sum\limits _{i=1} ^p \frac {rel_i}{log_2(i+1)}$

**Normalized Discounted Cumulated Gain(归一化折扣累积增益,nDCG):**由于搜索结果随着检索词的不同，返回的检索的数量是不一致的。DCG是一个累加的值，无法直接比较。nDCG就是用IDCG进行归一化处理，表示当前DCG比IDCG还差多大的距离。

$nDCG_p=\frac {DCG_p}{IDCG_p}$

其中 IDCG表示理想情形下DCG的最大值。用比值可以说明“现实”和“理想”的差距。

```python
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
        discounts = torch.log2(torch.arange(k).type(torch.FloatTensor) + 2)
        # 因为从0开始所以+2
        return torch.sum(gains / discounts)

    best = dcg_score(y_true, k, gains)
    actual = dcg_score(y_score, k, gains)
    result = actual / best
    # ndcg 就是用现有除以最优（实际）
    return result.item()
```



##平均精度均值(MAP)
#### **Precision(精确度,P)：**

精确度（Precision）是指检索中得到的文档中相关文档所占的比例。公式如下：
$$
precision=\frac {|retrieved\ documents|\cap |relevant\ documents|}{|retrieved\ documents|}
$$


precision@10代表10个文档中相关文档所占比例。

#### **Recall(召回率,R)：**

召回率是覆盖面的度量，是指从文档中，相关文档被检索到的比例。公式如下：
$$
precision=\frac {|retrieved\ documents|\cap |relevant\ documents|}{|relevant\ documents|}
$$
**Average precision(AveP)：**

由于召回率和精确率都只能衡量搜索模型的一个侧面。于是将准确率与召回率联立，把准确率看成是召回率的一个函数$P=f(R)$。那么就可以对函数P=f(R)P=f(R)在RR上进行积分，可以求PP的期望均值。AveP意义是在召回率从0到1逐步提高的同时，对每个R位置上的P进行相加，也即要保证准确率比较高，才能使最后的AveP比较大。

而通常我们会用多个查询词来衡量检索模型的性能。因而对多个查询语句的AveP求平均值，得到了**MAP，Mean average precision**。

##**综合评价指标（F-Measure）**
P和R指标有时候会出现的矛盾的情况，这样就需要综合考虑他们，最常见的方法就是F-Measure。
F-Measure是Precision和Recall加权调和平均：
$$
F=\frac {(a^2+1)P*R}{a^2(P+R)}
$$
a为可选参数。当a=1时，$F=\frac {2PR}{P+R}$

##ERR(Expected reciprocal rank)

#### **Mean reciprocal rank (MRR) ：**reciprocal rank指的是，第一个正确答案的排名的倒数。而MRR就指的是，多个查询语句的排名倒数的均值。

$$
MRR=\frac {1}{|Q|} \sum\limits _{i=1}^{|Q|}\frac{1}{rank_i}
$$

$rank_i$指的是第i个查询词中第一个正确答案的排名。

但是MRR不可导，无法作为梯度改进模型。因此Cascade Model（瀑布模型）出现，并运用于点击模型中。

#### Cascade Model（瀑布模型）：

**点击模型**中的瀑布模型，考虑到在同一个检索结果列表中各文档之间的位置依赖关系，假设用户从上至下查看，若是遇到某一检索结果项满意并进行点击，则操做结束；不然跳过该项继续日后查看。第 i 个位置的文档项被点击的几率为：
$$
P(C_i)=r_i\prod\limits_{j=1}^{i-1}(1-r_j)
$$
其中 $r_i$ 表示第 i 个文档被点击的几率，前 i-1 个文档则没有被点击，几率均为 $1-r_j$；

**ERR（Expected reciprocal rank)**

直译为预期的倒数排名。表示用户的需求被知足时中止的位置的倒数的期望。ERR是受到cascade model的启发。

用户在位置r处停止的概率$P_r$：
$$
P_r=R_r\prod\limits _{i=1}^{r-1}(1-R_i)
$$
与瀑布模型不同的是，此处的$R_i$是关于文档相关度等级的函数。设$R_i$为：
$$
R_i=\frac {2^{l_i}-1}{2^{l_m}}
$$
$l_m$代表相关性最高的一档，而$l_i$表示样本中第i个结果的相关性标记。

此时ERR计算为：
$$
ERR=\sum\limits_{r}\frac {1}{r}R_r\prod\limits _{i=1}^{r-1}(1-R_i)
$$
若将$\frac {1}{r}$换为$\frac{1}{log_2r+1}$此时与DCG相同。