#Ranking Learning Evaluation Indicators

##Introduction

Learning to Rank is to use machine learning methods to train a large number of ranking features on a data set, automatically learn parameters, and optimize evaluation indicators to generate a ranking model.

Ranking learning evaluation indicators are usually used to measure the performance of ranking models, such as accuracy (Precesion, P), recall (R), mean average precision (MAP),Cumulative gain (Normalized discounted cumulated gain, NDCG), Expected reciprocal rank (ERR), and normalized discounts commonly used in information retrieval. 

##Normalized discounted cumulated gain (NDCG)

NDCG is used as an evaluation index for the sorting result, and can directly evaluate the accuracy of the sorting. In nDCG, the relevance of documents can be scored in multiple levels.

The recommendation system usually returns a list of items for a user. Assuming that the length of the list is K, you can use NDCG@K to evaluate the gap between the sorted list and the user's actual interactive list.

The establishment of NDCG mainly follows two ideas: 　　

1. The result of high relevance will affect the final index score more than the result of general relevance;

2. The relevance result index at a higher position will be more affect;

NDCG stands for Normalized, discounted, and cumulated gain.

**Gain (gain): ** represents the relevance score of each item in the list, denoted as $rel_i$, which represents the relevance of this item at the position i.

```python
def rele_gain(rele_level, gain_base=2.0):
   gain = np.power(gain_base, rele_level) - 1.0
   return gain
```

**Cumulative Gain (CG): ** represents accumulating the gain of K items, denoted as $CG_p=\sum\limits _{i=1}^p rel_i$. Only the cumulative gain of the correlation degree is considered and the influence of the position is not considered.



**Discounted Cumulated Gain (DCG): **On the basis of CG, taking into account the factors of sorting order, the top-ranked item gains higher and the lower-ranked item is compromised. The purpose is to make the higher the ranking result affect the final result. This loss factor is $\frac{1}{log_2(i+1)}$.

So $DCG_p=\sum\limits _{i=1} ^p \frac {rel_i}{log_2(i+1)}$

**Normalized Discounted Cumulated Gain (nDCG): ** Since the search results vary with the search terms, the number of searches returned is inconsistent. DCG is an accumulated value and cannot be directly compared. nDCG uses IDCG for normalization, which indicates how far the current DCG is from IDCG.

$nDCG_p=\frac {DCG_p}{IDCG_p}$

Where IDCG represents the maximum value of DCG under ideal conditions. The ratio can illustrate the gap between "real" and "ideal".

```python
def tor_discounted_cumu_gain_at_ks(sorted_labels, max_cutoff, multi_level_rele=True):
    '''
    ICML-nDCG, which places stronger emphasis on retrieving relevant documents
    :param sorted_labels: ranked labels (either standard or predicted by a system) in the form of np array
    :param max_cutoff: the maximum rank position to be considered
    :param multi_lavel_rele: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
    :return: cumulative gains for each rank position
    '''
    
    if multi_level_rele:    #the common case with multi-level labels
        nums = torch.pow(2.0, sorted_labels[0:max_cutoff]) - 1.0
    else:
        nums = sorted_labels[0:max_cutoff]  #the case like listwise ranking, where the relevance is labeled as (n-rank_position)

    denoms = torch.log2(torch.arange(max_cutoff, dtype=torch.float) + 2.0)   #discounting factor
    dited_cumu_gains = torch.cumsum(nums/denoms, dim=0)   # discounted cumulative gain value w.r.t. each position

    return dited_cumu_gains
```
##Average Precision Mean (MAP)

#### **Precision (Precision, P):**

Precision (Precision) refers to the proportion of related documents in the documents obtained in the search. The formula is as follows:
$$
precision=\frac {|retrieved\ documents|\cap |relevant\ documents|}{|retrieved\ documents|}
$$

precision@10 represents the proportion of related documents in 10 documents.

```
""" Precision """
def tor_p_at_ks(sys_sorted_labels, ks=None):
    '''    precision at ks
    :param sys_sorted_labels: the standard labels sorted in descending order according to predicted relevance scores
    :param ks:
    :return:
    '''
    valid_max = sys_sorted_labels.size(0)
    used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    rele_ones = torch.ones(max_cutoff)
    non_rele_zeros = torch.zeros(max_cutoff)
    positions = torch.arange(max_cutoff) + 1.0

    sys_sorted_labels = sys_sorted_labels[0:max_cutoff]
    binarized_sys = torch.where(sys_sorted_labels > 0, rele_ones, non_rele_zeros)
    cumu_binarized_sys = torch.cumsum(binarized_sys, dim=0)

    sys_positionwise_precision = cumu_binarized_sys/positions
    sys_p_at_ks = sys_positionwise_precision[inds]
    if valid_max < max(ks):
        padded_p_at_ks = torch.zeros(len(ks))
        padded_p_at_ks[0:len(used_ks)] = sys_p_at_ks
        return padded_p_at_ks
    else:
        return sys_p_at_ks
```

#### **Recall (recall rate, R): **

Recall rate is a measure of coverage, which refers to the proportion of documents retrieved from related documents. The formula is as follows:
$$
precision=\frac {|retrieved\ documents|\cap |relevant\ documents|}{|relevant\ documents|}
$$
**Average precision(AveP):**

Because recall rate and accuracy rate can only measure one aspect of the search model. Therefore, the accuracy rate and the recall rate are combined, and the accuracy rate is regarded as a function of the recall rate $P=f(R)$. Then the function P=f(R)P=f(R) can be integrated on RR, and the expected mean value of PP can be obtained. The meaning of AveP is to add the P at each R position while the recall rate is gradually increasing from 0 to 1, that is, to ensure that the accuracy rate is relatively high, in order to make the final AveP relatively large.



```
""" Average Precision """
def tor_ap_at_ks(sys_sorted_labels, ks=None):
    ''' average precision at ks
    :param sys_sorted_labels: the standard labels sorted in descending order according to predicted relevance scores
    :param ks:
    :return:
    '''
    valid_max = sys_sorted_labels.size(0)
    used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    rele_ones = torch.ones(max_cutoff)
    non_rele_zeros = torch.zeros(max_cutoff)
    positions = torch.arange(max_cutoff) + 1.0

    sys_sorted_labels = sys_sorted_labels[0:max_cutoff]
    binarized_sys = torch.where(sys_sorted_labels > 0, rele_ones, non_rele_zeros)
    cumu_binarized_sys = torch.cumsum(binarized_sys, dim=0)

    sys_poswise_precision = cumu_binarized_sys / positions
    zeroed_sys_poswise_precision = torch.where(sys_sorted_labels>0, sys_poswise_precision, non_rele_zeros)    # for non-rele positions, use zero rather than cumulative sum of precisions

    cumsum_sys_poswise_precision = torch.cumsum(zeroed_sys_poswise_precision, dim=0)
    sys_poswise_ap = cumsum_sys_poswise_precision / positions
    sys_ap_at_ks = sys_poswise_ap[inds]

    if valid_max < max(ks):
        padded_ap_at_ks = torch.zeros(len(ks))
        padded_ap_at_ks[0:len(used_ks)] = sys_ap_at_ks
        return padded_ap_at_ks
    else:
        return sys_ap_at_ks
```

Usually we use multiple query terms to measure the performance of the retrieval model. Therefore, the AveP of multiple query statements is averaged, and **MAP, Mean average precision** is obtained.

##**Comprehensive Evaluation Index (F-Measure)**

P and R indicators sometimes have contradictory situations, so they need to be considered comprehensively. The most common method is F-Measure.
F-Measure is the weighted harmonic average of Precision and Recall:
$$
F=\frac {(a^2+1)P*R}{a^2(P+R)}
$$
a is an optional parameter. When a=1, $F=\frac {2PR}{P+R}$

##ERR(Expected reciprocal rank)

#### **Mean reciprocal rank (MRR): **The reciprocal rank refers to the inverse of the rank of the first correct answer. MRR refers to the average of the inverse rankings of multiple query statements.

$$
MRR=\frac {1}{|Q|} \sum\limits _{i=1}^{|Q|}\frac{1}{rank_i}
$$

$rank_i$ refers to the ranking of the first correct answer in the i-th query term.

However, MRR is not directable and cannot be used as a gradient improvement model. So the Cascade Model (waterfall model) appeared and used it in the click model.

#### Cascade Model:

The waterfall model in **click model**, taking into account the position dependence between the documents in the same search result list, assuming that the user views from top to bottom, if a certain search result item is satisfied and clicks, The operation is over; otherwise, skip this item and continue to view later. The probability of the document item at the i position being clicked is:
$$
P(C_i)=r_i\prod\limits_{j=1}^{i-1}(1-r_j)
$$
Among them, $r_i$ represents the probability that the i-th document is clicked, and the first i-1 document is not clicked, and the probability is $1-r_j$;

**ERR (Expected Reciprocal Rank)**

Literally translated as the expected inverse ranking. Represents the expectation of the inverse of the position where the user's needs are satisfied. ERR is inspired by the cascade model.

Probability $P_r$ of the user stops research at position r:
$$
P_r=R_r\prod\limits _{i=1}^{r-1}(1-R_i)
$$
Unlike the waterfall model, the $R_i$ here is a function of the document relevance level. Let $R_i$ be:
$$
R_i=\frac {2^{l_i}-1}{2^{l_m}}
$$
$l_m$ represents the most relevant file, and $l_i$ represents the correlation mark of the i-th result in the sample.

At this time ERR is calculated as:
$$
ERR=\sum\limits_{r}\frac {1}{r}R_r\prod\limits _{i=1}^{r-1}(1-R_i)
$$
If change $\frac {1}{r}$ to $\frac{1}{log_2r+1}$, this is the same as DCG.

```
""" ERR """
def tor_err_at_ks(sys_sorted_labels, ks=None, multi_level_rele=True, max_rele_level=None):
    '''
    :param sys_sorted_labels: the standard labels sorted in descending order according to predicted relevance scores
    :param ks:
    :param multi_level_rele:
    :param max_rele_level:
    :return:
    '''
    valid_max = sys_sorted_labels.size(0)
    used_ks = [k for k in ks if k <= valid_max] if valid_max < max(ks) else ks

    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)
    if multi_level_rele:
        positions = torch.arange(max_cutoff) + 1.0
        expt_ranks = 1.0 / positions    # expected stop positions

        tor_max_rele = torch.Tensor([max_rele_level]).float()
        satis_pros = (torch.pow(2.0, sys_sorted_labels[0:max_cutoff]) - 1.0)/torch.pow(2.0, tor_max_rele)
        non_satis_pros = torch.ones(max_cutoff) - satis_pros
        cum_non_satis_pros = torch.cumprod(non_satis_pros, dim=0)

        cascad_non_satis_pros = positions
        cascad_non_satis_pros[1:max_cutoff] = cum_non_satis_pros[0:max_cutoff-1]
        expt_satis_ranks = expt_ranks * satis_pros * cascad_non_satis_pros  # w.r.t. all rank positions
        err_at_ranks = torch.cumsum(expt_satis_ranks, dim=0)

        err_at_ks = err_at_ranks[inds]
        if valid_max < max(ks):
            padded_err_at_ks = torch.zeros(len(ks))
            padded_err_at_ks[0:len(used_ks)] = err_at_ks
            return padded_err_at_ks
        else:
            return err_at_ks
    else:
        raise NotImplementedError
```