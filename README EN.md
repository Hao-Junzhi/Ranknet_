#L2R:From RankNet Algorithm to LambdaMart Algorithm

##Introduction

RankNet is a pairwise Learning to Rank algorithm, the core of which is to solve the sorting problem from the perspective of probability. The ranking function is learned through the constructed probability loss function. 

##Algorithm implementation

The main core of the algorithm is to construct a probability loss function, as follows:

###**Correlation probability**:

The correlation probability is divided into the predicted correlation probability and the true correlation probability

####1.predicted correlation probability

> For any pair of doc pairs $P(U_i>U_j)$ , the model output score is $(s_i,s_j)$. According to the prediction of the model, the relevant probability $U_i$ is more relevant than $U_j$ is defined as:

![](http://latex.codecogs.com/gif.latex?\\P_{ij}=P(U_i>U_j)=\frac {1}{1+e^{-\sigma (s_i-s_j)}})

> Since the model used by RankNet is generally a neural network, the sigmoid function can provide a better probability assessment based on experience. σ is a learnable parameter, which determines the shape of the sigmoid function.

> RankNet has a inference: for any permutation of length n, only need to know the probability of n-1 adjacent items $P_{i,i+1}$, and you can infer any pairs without calculating all pairs. The sorting probability of each item. Given $P_{i,k}$ and $P_{k,j}$, $P_{i,j}$ can be derived through the following process. The mathematical proof is as follows:

![](http://latex.codecogs.com/gif.latex?\\P_{i,j}=\frac {1}{1+e^{-\sigma (s_i-s_j)}}\\
=\frac {1}{1+e^{-\sigma (s_i-s_k+s_k-s_j)}}\\
=\frac {e^{\sigma (s_i-s_k)}\cdot e^{-\sigma (s_k-s_j)}}{1+e^{\sigma (s_i-s_k)}\cdot e^{-\sigma (s_k-s_j)}}\\
=\frac {P_{i,k}\cdot P_{k,j}} {1+2P_{i,k} P_{k,j}-P_{i,k}-P_{k,j}})

 #### 2. Probability of true correlation

The doc of the pair in the training data has a label about query correlation to $(U_i, U_j)$. The meaning of this label is: whether $U_{i}$ is more related to query than $U_{j}$. Therefore, the true probability that $U_{i}$ is more relevant than $U_{j}$ is defined as follows:
$$
\overline{P_{ij}} = \frac{1+S_{ij}}{2}
$$
If $U_{i}$ is more relevant than $U_{j}$, then $S_{ij}=1$; if $U_{i}$ is not as relevant as $U_{j}$, then $S_{ij}= -1$; if $U_{i}$ and $U_{j}$ have the same degree of correlation, then $S_{ij}=0$.

### Construct the loss function

> For a ranking, RankNet evaluates the ranking result from the relative relationship of each doc. The better the ranking effect, the fewer pairs have wrong relative relationships. The wrong relative relationship is that $U_i$ is ranked before $U_j$ according to the model output, but the correlation of the true label $U_i$ is less than $U_j$, and then an error pair is recorded.

> RankNet essentially takes the minimum number of wrong pairs as the optimization goal. When abstracted as a cost function, RankNet actually introduces the idea of probability: instead of judging that $U_i$ is ranked in front of $U_j$, it is said that $U_i$ is ranked in front of $U_j$ with a certain probability P, that is The optimization goal is to minimize the gap between the predicted probability and the true probability. Finally, RankNet uses Cross Entropy as the cost function to measure the fit of $P_{i,j}$ to $\overline {P_{i,j}}$, which is defined as follows:

For the predicted correlation probability and the true correlation probability, use the cross entropy function as the loss function:
$$
C_{i,j}=-\overline {P_{i,j}}log(P_{i,j})-(1-\overline {P_{i,j}})log(1-P_{i,j})
$$
The cross entropy loss function (loss) of a single sample can be derived as:
$$
C_{ij}=1/2*(1-S_{ij})*σ(s_i-s_j)+log[1+e^{-σ(s_i-s_j)}]
$$
At this point, even if the model scores calculated by two documents with different correlations are the same ($s_{i}=s_{j}$), and the value of the loss function is greater than 0, the pair will still be penalized to make them The sort position is distinguished.

At this time, the loss function is differentiable.

batch_loss_1st = $0.5 * sigma( s_i-s_j) * (1.0 - S_{ij})$
batch_loss_2nd = torch.log(torch.exp(-sigma * si_sj) + 1.0)

The loss function is the sum of the loss functions of all samples under the same query.

###Training

After obtaining the differentiable cost function, the stochastic gradient descent method can be used to iteratively update the model parameter w, and update the weight once for each pair.

However, there is a problem with this, because when calculating the gradient, the weight is updated once for each pair. This time complexity is $O(n^2)$. Using mini-batch learning, all doc under the same query can be updated once. So the time complexity is $O(n)$. 

Derive as follows:

Use the stochastic gradient descent method to iteratively update the model parameters $w_k$:
$$
w_k\rightarrow w_k-\eta \frac{\partial C}{\partial w_k}
$$
When using the stochastic gradient descent method:
$$
\frac{\partial C}{\partial w_k} = \frac{\partial C}{\partial s_i} \frac{\partial s_i} 
{\partial w_k}+\frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k}\\
=\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}}) (\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\\
=\lambda _{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})
$$
Where $\lambda _{ij}$ =$\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}}) $

When switching to the gradient descent method using the mini-batch processing:
$$
\frac{\partial C}{\partial w_k}=\displaystyle \sum \frac{\partial C}{\partial s_i} \frac{\partial s_i} 
{\partial w_k}+\frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k}
$$

$$
\frac{\partial C_{ij}}{\partial s_i}=\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}})=-\frac{\partial C_{ij}}{\partial s_j}
$$

Let $\lambda _{ij}$ =$\sigma (0.5(1-S_{ij})+\frac {1}{1+e^{\sigma (s_i-s_j)}}) $
$$
\frac{\partial C}{\partial w_k} =\displaystyle \sum\lambda _{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\\
=\displaystyle \sum \lambda _i \frac{\partial s_i}{\partial w_k}
$$

$$
\lambda _i=\sum\limits_{i,j\in I} \lambda _{ij}-\sum\limits_{j,i\in I} \lambda _{ij}
$$

$λ_i$ determines the direction and amplitude of the movement of the i-th doc in the iteration. The fewer doc that is actually ranked in front of $U_i$ and the more doc that is ranked behind $U_i$, the greater the extent of the document $U_i$ moving forward (actual $λ_i$ The more negative the more move forward). This indicates that the direction and strength of the next reordering of each f depends on the documents with different labels that can form the "pair" of the relative relevance judgment under the same Query.

But this also determines that the direction and range of movement cannot be determined according to the desired direction such as NDCG and other search indicators. Thus the LambdaRank algorithm was born as the times require.

##LambdaRank Algorithm

Evaluation indicators in information retrieval such as NDCG only focus on the ranking of the top k results. Because these indicators are not derivable or derivatives do not exist, when we adopt the RankNet algorithm, we often cannot iterate with them as the optimization objective (loss function), so there is still a gap between the optimization objective of RankNet and the information retrieval evaluation indicator.

![image](https://github.com/Hao-Junzhi/Ranknet_/raw/master/images/lambda.png)

> As shown in the figure above, each line represents a document, blue represents related documents, and gray represents irrelevant documents. RankNet calculates the cost in a pairwise error method. The cost on the left is 13, and the right is reduced by 3 for the first related document. There are 5 positions in the second document, and the cost is reduced to 11. However, evaluation indicators such as NDCG or ERR only focus on the ranking of the top k results. In the optimization process, lowering the position of the previous related documents is not what we want The results obtained. The black arrow on the left of the right picture of Figure 1 indicates the direction and intensity of RankNet's next round of ordering, but what we really need is the direction and intensity represented by the red arrow on the right, that is, we pay more attention to the improvement of the ranking position of related documents in the front position. LambdaRank evolved based on this idea. Lambda refers to the red arrow, which represents the direction and intensity of the next iteration optimization, which is the gradient.

LambdaRank is an empirical algorithm. It does not solve the ranking problem by displaying the definition of the loss function and then finding the gradient, but analyzes the physical meaning of the gradient required by the ranking problem, and directly defines the gradient, that is, the Lambda gradient. We need to use nDCG to adjust the direction and intensity of the ordering of each iteration.

Specifically, it is necessary to improve the existing loss or the gradient of the loss. And indicators such as NDCG are not diversified, so we skip loss and simply and roughly multiply the gradient in the form of the RankNet acceleration algorithm to define a new Lambda gradient.
$$
λ_{ij}=\frac{\Delta Z_{ij}}{1+e^{-\sigma (s_i-s_j)}}
$$
The evaluation index Z can be NDCG, ERR, etc.

The gradient of the loss function represents the direction and strength of the next iteration of the document optimization. Due to the introduction of an evaluation index that pays more attention to the correctness of the head, the Lambda gradient can further improve the ranking position of the high-quality documents in the front. This effectively avoids the situation that the position of the high-quality documents ranked higher is lowered. The advantage of LambdaRank over RankNet is that the training speed becomes faster after the factor is decomposed, and the evaluation index is considered at the same time to directly solve the problem, and the effect is more obvious.

