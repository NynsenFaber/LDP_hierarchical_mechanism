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
        :param eps: privacy parameter
        :param protocol: the protocol to use
        """
        super().__init__(B, b)

    def update_tree(self, data: list[Union[float, int]],
                    eps: float, protocol: str):
        servers, counts = ldp_protocol(data=data,
                                       eps=eps,
                                       tree=self,
                                       protocol=protocol)
        for i, level_attributes in enumerate(self.attributes):
            for j in range(len(level_attributes)):
                self.attributes[i][j] = get_frequency(servers[i], counts[i], j)
        self.N = sum(counts)


def get_frequency(server, count, item) -> float:
    """
    Estimate the frequency of an item using the server and the count.
    :param server: a server (an instance of LDP Frequency Oracle server of pure_ldp package)
    :param count: the count of the data (server returns absolute frequency)
    :param item: the item to estimate
    """
    return server.estimate(item, suppress_warnings=True) / count


# test
B = 8
b = 2
eps = 10
protocol = 'unary_encoding'
tree = Private_Tree(B, b)
data = np.random.randint(0, B, 100000)
tree.update_tree(data, eps, protocol)
cdf = tree.compute_cdf()
print(f"CDF: {cdf}")
# find closest index of the CDF to 0.5
closest_item = min(range(len(cdf)), key=lambda i: abs(cdf[i] - 0.5))
print(f"Closest item to 0.5: {closest_item}")
