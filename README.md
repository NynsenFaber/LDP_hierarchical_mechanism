# Hierarchical Mechanism for LDP
it is an implementation of the Hierarchical Mechanism in

> Kulkarni, Tejas, Graham Cormode, and Divesh Srivastava. "Answering range queries under local differential privacy." arXiv preprint arXiv:1812.10942 (2018).

The LDP frequency protocol are implemented from the library 
https://github.com/Samuel-Maddock/pure-LDP

The library is intended for testing the LDP hierarchical mechanism locally.
## Install

    pip install hierarchical-mechanism-LDP

## Usage
It is based on the class `Private_Tree` that implements the hierarchical mechanism for local differential privacy. The class has the following methods:
### Initialization

```python
# Initialization

from hierarchical_mechanism_LDP import Private_TreeBary
import numpy as np

B = 4000  # bound of the data, i.e., the data is in [0, B]
b = 4  # branching factor of the tree
eps = 1  # privacy budget
q = 0.4  # quantile to estimate
protocol = 'unary_encoding'  # protocol to use for LDP frequency estimation

tree = Private_TreeBary(B, b)

data = np.random.randint(0, B, 100000)  # generate random data

tree.update_tree(data, eps, protocol, post_process=True)  # update the tree with the data
```
The `post_process` boolean parameter is used to apply the post-processing step of the hierarchical mechanism.
The post-processing step is used to improve the accuracy of the mechanism. The post-processing step is applied by default.

Post-processing applies the Hierarchial Mechanism proposed by Hay et al. in 
> M. Hay, V. Rastogi, G. Miklau, and D. Suciu. Boosting the
accuracy of differentially private histograms through
consistency. PVLDB, 3(1):1021–1032, 2010.
### Quantile estimation
You can estimate the quantile of the data with `Private_Tree.get_quantile(q)`, where `q` is the quantile to estimate.
```python
# get quantile of the data
true_quantile = np.quantile(data, q)
private_quantile = tree.get_quantile(q)  # get the quantile
print(f"DP quantile: {private_quantile}")
print(f"True quantile: {true_quantile}")

```
Result
```
Private quantile: 1591
True quantile: 1598.0
```

### Range Queries
You can estimate the range queries of the data with `Private_Tree.get_range_query(a, b)`, where `a` and `b` are the bounds of the range query.
Additionally, you can return a normalized range query.

```python

left = 1000
right = 2000
true_range_query = np.sum(data >= left) - np.sum(data >= right)
private_range_query = tree.get_range_query(left, right, normalized=False)
print(f"True range query: {true_range_query}")
print(f"Private range query: {private_range_query}")
```
Result
```
True range query: 24980
Private range query: 25514.970123615636
```
### Binning
Given a list of quantiles and an error `alpha`, you can create bins of the form 
`[q_i - alpha, q_i + alpha]` with `Private_Tree.get_bins(quantiles, alpha)`. The bins
are returned as a list of tuples.
```python
# test binning
quantiles = [0.25, 0.50, 0.75]
alpha = 0.1
bins = tree.get_bins(quantiles, alpha)
print(bins)
```
Result
```
[(np.int64(664), np.int64(1441)), (np.int64(1591), np.int64(2562)), (np.int64(2669), np.int64(3365))]
```