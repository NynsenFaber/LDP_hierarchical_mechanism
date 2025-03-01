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

N = 10_000
B = 4000  # bound of the data, i.e., the data is in [0, B]
b = 4  # branching factor of the tree
eps = 5.  # privacy budget
q = 0.4  # quantile to estimate
data = np.random.randint(0, B, N)  # generate random data

tree = Private_TreeBary(B, b, eps, on_all_levels=True)

tree.update_tree(data)  # update the tree with the data
```
With the `Private_Tree` class, you can create a hierarchical mechanism with the following parameters:
- `B`: the bound of the data, i.e., the data is in [0, B]
- `b`: the branching factor of the tree
- `eps`: the privacy budget
- `on_all_levels`: boolean parameter that decides if the users reports to all levels or they sample a random level to report. The default value is `True`.

Other parameters of `Private_Tree` are:
- `protocol`: It set to default value "unary encoding" for unary encoding protocol for small dimesions. At large dimension >10_000 the protocol is set to be "Hadamard Randomized Response".

### Additional Features
```python
tree.compute_attributes()  # compute the attributes of the tree and delete the frequency protocol
tree.post_process()  # It post-process the tree attributes and compute the entire cdf.
```
With the `compute_attributes` method, all the frequency oracles are used to estimate the tree attributes. The tree attributes are the number of users in each node of the tree. The frequency protocol is deleted after the computation of the attributes.

The `post_process` method is used to apply the post-processing step of the hierarchical mechanism.
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

### Shuffling Amplification
We implemented the algorithm provided in: 
> Feldman, Vitaly, Audra McMillan, and Kunal Talwar.
> "Hiding among the clones: A simple and nearly optimal analysis of privacy amplification by shuffling." 2021 IEEE 
> 62nd Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 2022.

For analyzing the privacy amplification by shuffling of the mechanism. The
implementation is taken from the repository
https://github.com/apple/ml-shuffling-amplification

```python
# Test amplification by shuffling
delta = 1e-6
shuffle_numerical = tree.get_privacy(shuffle=True, delta=delta, numerical=True)
shuffle_theoretical = tree.get_privacy(shuffle=True, delta=delta, numerical=False)
print(f"For an initial {tree.eps}-DP mechanism, after shuffling {tree.N} users and considering delta = {delta} we obtain"
      f"a numerical upper bound of eps = : {shuffle_numerical} and a theoretical upper bound of eps = {shuffle_theoretical}")
```
Result
```
For an initial 1-DP mechanism, after shuffling 100000 users and considering delta = 1e-06 
we obtain a numerical upper bound of eps = : 0.015513881255146383 
and a theoretical upper bound of eps = 0.07529011566478624
```