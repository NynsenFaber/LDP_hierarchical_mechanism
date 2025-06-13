from hierarchical_mechanism_LDP.mechanism import sum_chunks, Private_TreeBary
import numpy as np


def test_sum_chunks():
    X = np.array([1, 1, 1, 2, 2, 2])
    assert np.allclose(sum_chunks(X, 3), [3, 6])

    X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert np.allclose(sum_chunks(X, 3), [3, 6, 9])


def test_Private_TreeBary():
    print("Testing Private Tree Bary")

    # test Quantile
    B = int(10 ** 4)
    b = 2
    eps = 1.
    N = 20_000
    protocol = 'hadamard_mechanism'

    print("Testing with")
    print(f"B = {B}, b = {b}, eps = {eps}, N = {N}, protocol = {protocol}\n")
    tree = Private_TreeBary(B, b, eps, on_all_levels=True, protocol=protocol)
    data = np.random.randint(1, B, N)
    # get private quantile
    tree.update_tree(data)
    # post process the tree using algorithm Hay et al. 2009 so to have a consistent tree
    tree.post_process(delete_attributes=False)
    # note that `delete_attributes=False` is used to keep the attributes for further checks,
    # in production you might want to set it to `True`
    # checks consistency of the tree
    for level in range(0, tree.depth - 1):
        children_sum = sum_chunks(np.array(tree.attributes[level + 1]), tree.b)
        assert np.allclose(tree.attributes[level], children_sum), f"Level {level} is not consistent"
    # get median
    private_median = tree.get_quantile(0.5)
    # get rank error
    rank_private_median = data.searchsorted(private_median)
    print("Private median:", private_median)
    print("True median:", np.median(data))
    print("CDF of private median:", rank_private_median / len(data))
    print("Quantile Error:", abs(rank_private_median / len(data) - 0.5))
    print("Privacy budget LDP:", tree.privacy_accountant())
    print("Privacy budget Shuffle:", tree.privacy_accountant(shuffle=True, numerical=True, delta=1e-6))


test_sum_chunks()
test_Private_TreeBary()
