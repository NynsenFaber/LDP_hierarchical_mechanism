# Hierarchical Mechanism for LDP
Contained in `hierarchical_mechanism_LDP` it is an implementation of the Hierarchical Mechanism in

*Kulkarni, Tejas, Graham Cormode, and Divesh Srivastava. "Answering range queries under local differential privacy." arXiv preprint arXiv:1812.10942 (2018).

It gives the cumulative distribution function of the data in a local differentially private way.
### Usage
```python
from hierarchical_mechanism_LDP import Private_Tree
import numpy as np

B = 4000
b = 4
eps = 1
q = 0.4
protocol = 'unary_encoding'
tree = Private_Tree(B, b)
data = np.random.randint(0, B, 100000)
# get quantile of the data
true_quantile = np.quantile(data, q)
# get private quantile
tree.update_tree(data, eps, protocol)  # update the tree with the data
tree.compute_cdf()  # compute the cdf of the data
private_quantile = tree.get_quantile(q)  # get the quantile
print(f"Closest item to {q}: {private_quantile}")
print(f"True quantile: {true_quantile}")

```
Result
```
Private quantile: 1591
True quantile: 1598.0
```