from hierarchical_mechanism_LDP.mechanism import sum_chunks, Private_TreeBary
import numpy as np


def test_sum_chunks():
    X = np.array([1, 1, 1, 2, 2, 2])
    assert np.allclose(sum_chunks(X, 3), [3, 6])

    X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert np.allclose(sum_chunks(X, 3), [3, 6, 9])


def test_Private_TreeBary():
    # test Quantile
    B = 154
    b = 4
    eps = 1.
    protocol = 'unary_encoding'
    tree = Private_TreeBary(B, b, eps)
    data = np.random.randint(0, B, 1000)
    # get private quantile
    tree.update_tree(data, post_process=False, delete_data=True)
    tree.post_process()
    # checks
    for level in range(0, tree.depth - 1):
        children_sum = sum_chunks(np.array(tree.attributes[level + 1]), tree.b)
        assert np.allclose(tree.attributes[level], children_sum), f"Level {level} is not consistent"

    # test Quantile
    B = 1545
    b = 5
    eps = 1.
    protocol = 'unary_encoding'
    tree = Private_TreeBary(B, b, eps)
    data = np.random.randint(0, B, 1000)
    # get private quantile
    tree.update_tree(data, post_process=False, delete_data=True)
    tree.post_process()
    # checks
    for level in range(0, tree.depth - 1):
        children_sum = sum_chunks(np.array(tree.attributes[level + 1]), tree.b)
        assert np.allclose(tree.attributes[level], children_sum), f"Level {level} is not consistent"


test_sum_chunks()
test_Private_TreeBary()
