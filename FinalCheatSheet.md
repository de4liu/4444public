
# IDSC 4444 Final Exam Formula Sheet
## Association Rules
**Support count** of an item set $X$: 
$$\sigma(X)= \text{Count}(X)$$
**Support percentage** of an item set $X$: 
$$\text{Supp}(X)= \frac{\text{Count}(X)}{\text{Num of Transactions}}$$
**Support count** of an association rule $X\rightarrow Y$
$$\sigma(X\rightarrow Y)= \text{Count}(X\ \text{and}\ Y)$$
**Support percentage** of an association rule $X\rightarrow Y$: 
$$\text{Supp}(X\rightarrow Y)= \frac{\text{Count}(X\ \text{and}\ Y)}{\text{Num of Transactions}}$$

**Confidence** of $X\rightarrow Y$:

$$\text{Conf}(X \rightarrow Y) = \frac{\sigma(X \rightarrow Y)}{\sigma(X)}  =  \frac{\text{Count}(X \text{ and } Y)}{\text{Count}(X)}$$

**Lift** of $X\rightarrow Y$:

$$\text{Lift}(X \rightarrow Y) = \frac{\text{Supp}(X \rightarrow Y)}{\text{Supp}(X)\text{Supp}(Y)}  = \frac{\text{Conf}(X \rightarrow Y)}{\text{Supp}(Y)}$$

## Clustering

Denote two observations as $(a_1, a_2, ..., a_k)$ and $(b_1, b_2, ..., b_k)$,

**Euclidian Distance**	
$$d(A,B) =	\sqrt{(a_1 − b_1)^2 +  (a_2 − b_2)^2 + ... + (a_k − b_k)^2}$$
**Manhattan Distance**
$$d(A,B) =	|a_1 − b_1| +  |a_2 − b_2| + ... + |a_k − b_k| $$


Let k be the number of binary features, $N_{01}$ be the number of features where observation 1 has a value of 0 and observation 2 has a value of 1; $N_{00}$, $N_{10}$, $N_{11}$ are similarly defined). 

**Matching Distance**	
$$d(A, B)=(N_{01}+N_{10})/(N_{00}+N_{01}+N_{10}+N_{11}) = (N_{01}+N_{10})/k$$

**Jaccard Distance**
$$d(A, B)=(N_{01}+N_{10})/(N_{01}+N_{10}+N_{11})$$


**Min-max Normalization**: 
$$z = \frac{x – \min}{\max – \min}$$

**Standardization (Z-score transformation)**: 

$$z =\frac{X – \text{Sample Mean}}{\text{Sample Standard Deviation}}$$

**Within Sum of Squared Errors (WSS)**:

$$\sum_{i=1}^K\sum_{x \in C_i}^K \text{d}(x,m_i)^2$$


where $\text{d}(x, m_i)$ is the Euclidian distance between each point $x$ in the cluster $C_i$ and the centroid $m_i$ of the cluster . The distances are squared and summed for each cluster; the total WSS is then sum of the WSS for all clusters.

**Between Sum of Squared Errors (BSS)**:

$$\sum_{i=1}^Kn_i \text{d}(m_i,m^\ast)^2$$

where $d(m_i, m^\ast)$ is the Euclidian distance between each cluster’s centroid $m_i$ and the centroid of the entire dataset $m^\ast$. The distances are then squared and summed together.

## Classification

**Entropy**
 
$$ \text{Entropy} = - \sum_{k=1}^m p_k \log(p_k) = -\left[p_1\log(p_1) + p_2\log(p_2) + ... p_m\log(p_m) \right]$$
 
where $m$ is the number of classes and $p$ is the prob. of the class.

**Information Gain**

Consider a split of a set of outcomes $S$ into two subsets $S_1$ and $S_2$. Let $p_1$ and $p_2$ be the proportion of $S_1$ and $S_2$ respectively. 

$$ \text{Information Gain} = \text{Entropy}({S}) - (p_1\text{Entropy}({S_1})+p_2\text{Entropy}({S_2})) $$


Let:
- $FN$ be the number of True Positives
- $FP$ be the number of False Positives
- $TN$ be the number of True Negatives
- $FN$ be the number of False Negatives

$$ \text{Error Rate}= \frac{FN + FP}{TP + TN + FP + FN}$$

$$ \text{Accuracy} = 1- \text{Error Rate}  =  \frac{TP + TN}{TP + TN + FP + FN}$$

$$ \text{Precision of the positive class (Positive predicted value)} =	 \frac{TP}{TP + FP}$$

$$ \text{Precision of the negative class (Negative predicted value)} =	 \frac{TN}{TN + FN}$$

$$ \text{Recall of the positive class (Sensitivity)} =	 \frac{TP}{TP + FN}$$

$$ \text{Recall of the negative class (Specificity)} =	 \frac{TN}{TN + FP}$$

$$\text{F-1 Score (can be computed for either class)} = 2 \times \frac{\text{Recall}\times\text{Precision}}{\text{Recall} + \text{Precision}}$$


## Numerical Prediction (Regression)


$$\text{Prediction Error } (e_i) = \text{Predicted Value } (p_i) – \text{Actual Value } (a_i)$$


$$\text{Mean Error (ME)} = \frac{1}{n}\sum_{i=1}^n e_i$$

$$\text{Mean Absolute Error (MAE)} = \frac{1}{n}\sum_{i=1}^n |e_i|$$

$$\text{Mean Absolute Percentage Error (MAPE)} = 100\times \frac{1}{n}\sum_{i=1}^n |\frac{e_i}{a_i}| $$

$$\text{Root Mean Square Error (RMSE)} = \sqrt{\frac{1}{n}\sum_{i=1}^n e_i^2} $$

$$\text{Sum of Squared Deviations (SSD) or Total Sum of Squares} = \sum_{i=1}^n e_i^2 $$
