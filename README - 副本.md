# RankNet 算法解析

## 简介

RankNet是一种pairwise的Learning to Rank算法，核心是通过概率的角度来解决排序问题。通过构造的概率损失函数来进行排序函数的学习。一般用神经网络方法进行实现。

## 算法实现

该算法的主要核心为构建概率损失函数，具体如下：

###（一）相关性概率

相关性概率分为预测的相关性概率与真实的相关性概率

#### 1.预测相关性概率

> 对于任意的一个pair $P(U_i>U_j)$ 对于任意一个pair的doc对,模型输出的score为$(s_i,s_j)$，那么根据模型的预测，$U_i$比$U_j$与query更相关的概率定义为

$$
P_{ij} = P(U_i>U_j) = \frac{1}{1+e^{-\sigma (s_i-s_j)}}
$$

> 由于RankNet使用的模型一般为神经网络，根据经验sigmoid函数能提供一个比较好的概率评估。σ为可学习参数,决定了sigmoid函数的形状。

> RankNet有一个结论：对于任何一个长度为n的排列，只需要知道n-1个相邻item的概率$P_{i,i+1}$ ，不需要计算所有的pair，就可以推断出来任何两个item的排序概率。已知$P_{i,k}$和$P_{k,j}$，$P_{i,j}$则可通过下面的过程推导得出。数学证明如下：

$$
P_{i,j}=\frac {1}{1+e^{-\sigma (s_i-s_j)}}\\
=\frac {1}{1+e^{-\sigma (s_i-s_k+s_k-s_j)}}\\
=\frac {e^{\sigma (s_i-s_k)}\cdot e^{-\sigma (s_k-s_j)}}{1+e^{\sigma (s_i-s_k)}\cdot e^{-\sigma (s_k-s_j)}}\\
=\frac {P_{i,k}\cdot P_{k,j}} {1+2P_{i,k} P_{k,j}-P_{i,k}-P_{k,j}}
$$



#### 2.真实相关性概率

> 训练数据中的pair的doc对$(U_i, U_j)$有一个关于query相关性的label，该label含义为：$U_{i}$比$U_{j}$与query更相关是否成立。因此，定义$U_{i}$比$U_{j}$更相关的真实概率如下：

$$
\overline{P_{ij}} = \frac{1+S_{ij}}{2}
$$

> 如果$U_{i}$比$U_{j}$更相关，则$S_{ij}=1$；如果$U_{i}$不如$U_{j}$相关，则$S_{ij}=-1$；如果$U_{i}$和$U_{j}$相关程度相同，则$S_{ij}=0$。

### (二)构建损失函数

> 对于一个排序，RankNet从各个doc的相对关系来评价排序结果的好坏，排序的效果越好，那么有错误相对关系的pair就越少。所谓错误的相对关系即如果根据模型输出<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>排在<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，但真实label为<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>的相关性小于<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>，那么就记一个错误pair。

> RankNet本质上就是以错误的pair最少为优化目标。而在抽象成cost function时，RankNet实际上是引入了概率的思想：不是直接判断<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>排在<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，而是说<a href="https://www.codecogs.com/eqnedit.php?latex=U_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{i}" title="U_{i}" /></a>以一定的概率P排在<a href="https://www.codecogs.com/eqnedit.php?latex=U_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?U_{j}" title="U_{j}" /></a>前面，即是以预测概率与真实概率的差距最小作为优化目标。最后，RankNet使用Cross Entropy作为cost function，来衡量<a href="https://www.codecogs.com/eqnedit.php?latex=P_{i,j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{i,j}" title="P_{i,j}" /></a>对<a href="https://www.codecogs.com/eqnedit.php?latex=\overline{P_{i,j}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\overline{P_{i,j}}" title="\overline{P_{i,j}}" /></a>的拟合程度，定义如下：

对于预测相关性概率和真实相关性概率，使用交叉熵函数作为损失函数
$$
C_{i,j}=-\overline {P_{i,j}}log(P_{i,j})-(1-\overline {P_{i,j}})log(1-P_{i,j})
$$


单个样本的交叉熵损失函数（loss）可以推导为
$$
C_{ij}=1/2*(1-S_{ij})*σ(s_i-s_j)+log[1+e^{-σ(s_i-s_j)}]
$$
此时，即使两个相关性不同的文档算出来的模型分数相同时（$s_{i}=s_{j}$），损失函数的值大于0，仍会对这对pair做惩罚，使他们的排序位置区分开。

此时损失函数和可微。

batch_loss_1st = $0.5 * sigma( s_i-s_j) * (1.0 - S_{ij})$
batch_loss_2nd = torch.log(torch.exp(-sigma * si_sj) + 1.0)

损失函数为同一query下所有样本的损失函数的总和

### (三)训练

得到可微的代价函数后，可以用随机梯度下降法来迭代更新模型参数w，对于每一对pair进行一次权重的更新。

但是这样有一个问题，由于计算梯度的时候，是对每一个pair对都会进行一次权重的更新。这样的时间复杂度为$O（n^2）$。采用mini-batch learning的方式，可以对同一个query下的所有doc进行一次权重的更新。这样时间复杂度就为$O(n)$。大大降低了反向传播的次数

推导如下：

用随机梯度下降法来迭代更新模型参数wk：
$$
w_k\rightarrow w_k-\eta \frac{\partial C}{\partial w_k}
$$
 

当使用随机梯度下降法的时候：
$$
\frac{\partial C}{\partial w_k} = \frac{\partial C}{\partial s_i} \frac{\partial s_i} 
{\partial w_k}+\frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k}\\
=\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}}) (\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\\
=\lambda _{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})
$$
其中$\lambda _{ij}$ =$\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}}) $



当转为利用批处理的梯度下降法：
$$
\frac{\partial C}{\partial w_k}=\displaystyle \sum \frac{\partial C}{\partial s_i} \frac{\partial s_i} 
{\partial w_k}+\frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k}
$$
其中
$$
\frac{\partial C_{ij}}{\partial s_i}=\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}})=-\frac{\partial C_{ij}}{\partial s_j}
$$
我们让$\lambda _{ij}$ =$\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}}) $

得到
$$
\frac{\partial C}{\partial w_k} =\displaystyle \sum\lambda _{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\\
=\displaystyle \sum \lambda _i \frac{\partial s_i}{\partial w_k}
$$


其中 
$$
\lambda _i=\sum\limits_{i,j\in I} \lambda _{ij}-\sum\limits_{j,i\in I} \lambda _{ij}
$$

> λi决定着第i个doc在迭代中的移动方向和幅度，真实的排在Ui前面的doc越少，排在Ui后面的doc越多，那么文档Ui向前移动的幅度就越大(实际λi负的越多越向前移动)。这表明每个f下次调序的方向和强度取决于同一Query下可以与其组成relative relevance judgment的“pair对”的其他不同label的文档。

但是这也决定了无法按照想要的方向如NDCG等检索指标去决定移动方向和幅度。因而诞生了LambdaRank算法。



## LambdaRank算法

NDCG等信息检索中的评价指标只关注top k个结果的排序。由于这些指标不可导或导数不存在，当我们采用RankNet算法时，往往无法以它们为优化目标（损失函数）进行迭代，所以RankNet的优化目标和信息检索评价指标之间还是存在差距。

![image](https://github.com/Hao-Junzhi/Ranknet_/raw/master/images/lambda.png)

> 如上图所示，每个线条表示文档，蓝色表示相关文档，灰色表示不相关文档，RankNet以pairwise error的方式计算cost，左图的cost为13，右图通过把第一个相关文档下调3个位置，第二个文档上条5个位置，将cost降为11，但是像NDCG或者ERR等评价指标只关注top k个结果的排序，在优化过程中下调前面相关文档的位置不是我们想要得到的结果。图 1右图左边黑色的箭头表示RankNet下一轮的调序方向和强度，但我们真正需要的是右边红色箭头代表的方向和强度，即更关注靠前位置的相关文档的排序位置的提升。LambdaRank正是基于这个思想演化而来，其中Lambda指的就是红色箭头，代表下一次迭代优化的方向和强度，也就是梯度。

LambdaRank是一个经验算法，它不是通过显示定义损失函数再求梯度的方式对排序问题进行求解，而是分析排序问题需要的梯度的物理意义，直接定义梯度，即Lambda梯度。我们需要利用nDCG，调整每一轮迭代的调序的方向与强度。

具体来说，由于需要对现有的loss或loss的梯度进行改进。而NDCG等指标又不可导，我们便跳过loss，直接简单粗暴地在RankNet加速算法形式的梯度上再乘一项，以此新定义了一个Lambda梯度

$λ_{ij}=\frac{\Delta Z_{ij}}{1+e^{-\sigma (s_i-s_j)}}$

评价指标Z可以为NDCG、ERR等。

损失函数的梯度代表了文档下一次迭代优化的方向和强度，由于引入了更关注头部正确率的评价指标，Lambda梯度得以让位置靠前的优质文档排序位置进一步提升。有效避免了排位靠前的优质文档的位置被下调的情况发生。LambdaRank相比RankNet的优势在于分解因式后训练速度变快，同时考虑了评价指标，直接对问题求解，效果更明显。