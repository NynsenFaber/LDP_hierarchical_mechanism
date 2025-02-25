from ldp_protocol import ldp_protocol
from data_structure import Tree
from typing import Union
import numpy as np


class Private_Tree(Tree):

    def __init__(self, B: int, b: int):
        """
        Constructor

        :param B: bound of the data
        :param b: branching factor of the tree
        """
        super().__init__(B, b)
        # attributes have the same shape of intervals but initialized with zeros
        self.attributes: list[list[list[int]]] = [[0] * len(interval) for interval in self.intervals]
        self.N = None  # total number of users that updated the tree
        self.cdf = None

    def update_tree(self, data: list[Union[float, int]],
                    eps: float, protocol: str):

        # run the LDP protocol
        servers, counts = ldp_protocol(data=data,
                                       eps=eps,
                                       tree=self,
                                       protocol=protocol)

        # update the attributes of the Private tree, do not update the root
        for i, level_attributes in enumerate(self.attributes):
            if i == 0: # the root gets 1.
                self.attributes[i] = [1.]
                continue
            for j in range(len(level_attributes)):
                # as the root is not updated, we need to shift the index by 1
                self.attributes[i][j] = get_frequency(servers[i - 1], counts[i - 1], j)
        self.N = sum(counts)

    def get_range_query(self, left: int, right: int, normalized: bool = False):
        assert 0 <= left <= right <= self.B, "Left and right must be between 0 and B"

        # compute right quantile
        indices = self.get_bary_decomposition_index(right)
        result_right = 0
        for i, j in indices:
            # attributes are normalized so we are summing frequencies
            result_right += self.attributes[i][j]

        # compute left quantile
        indices = self.get_bary_decomposition_index(left)
        result_left = 0
        for i, j in indices:
            # attributes are normalized so we are summing frequencies
            result_left += self.attributes[i][j]
        if normalized:
            return result_right - result_left
        else:
            return (result_right - result_left) * self.N

    def compute_cdf(self):
        """
        Compute the CDF of [0, b^(depth + 1)]

        :return:
        """
        cdf = np.zeros(self.b ** self.depth + 1)
        for i in range(self.b ** self.depth + 1):
            indices = self.get_bary_decomposition_index(i)
            cdf[i] = sum(self.attributes[j][k] for j, k in indices)
        self.cdf = cdf

    def get_quantile(self, quantile: float):
        """
        Get the quantile of the data

        :param quantile: the quantile to get

        :return: the quantile
        """
        assert 0 <= quantile <= 1, "Quantile must be between 0 and 1"

        if self.cdf is None:
            self.compute_cdf()
        # retrive only elements with positive values
        index = np.where(self.cdf - quantile >= 0)[0]
        # find the minimum index that is closest to the quantile
        return min(index, key=lambda i: self.cdf[i] - quantile)


def get_frequency(server, count, item) -> float:
    """
    Estimate the frequency of an item using the server and the count.
    :param server: a server (an instance of LDP Frequency Oracle server of pure_ldp package)
    :param count: the count of the data (server returns absolute frequency)
    :param item: the item to estimate
    """
    return server.estimate(item, suppress_warnings=True) / count


# test Quantile
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
tree.update_tree(data, eps, protocol)
tree.compute_cdf()
private_quantile = tree.get_quantile(q)
print(f"Closest item to {q}: {private_quantile}")
print(f"True quantile: {true_quantile}")

# test range query
left = 1000
right = 2000
true_range_query = np.sum(data >= left) - np.sum(data >= right)
private_range_query = tree.get_range_query(left, right, normalized=False)
print(f"True range query: {true_range_query}")
print(f"Private range query: {private_range_query}")
