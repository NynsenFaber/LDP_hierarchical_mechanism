from .ldp_protocol import ldp_protocol
from .data_structure import TreeBary
from typing import Union
import numpy as np


class Private_TreeBary(TreeBary):

    def __init__(self, B: int, b: int):
        """
        Constructor

        :param B: bound of the data
        :param b: branching factor of the tree
        """
        super().__init__(B, b)
        # attributes have the same shape of intervals but initialized with zeros
        self.attributes: list[list[float]] = [[0.] * len(interval) for interval in self.intervals]
        self.N = None  # total number of users that updated the tree
        self.cdf = None

    def __getitem__(self, item: tuple[int, int]) -> float:
        """
        Get the attribute at the given level and index

        :param item: the level and index of the interval

        :return: the attribute at the given level and index
        """
        return self.attributes[item[0]][item[1]]

    def update_tree(self, data: list[Union[float, int]],
                    eps: float, protocol: str, post_process: bool = True):
        """
        Update the tree with the data using the LDP protocol. If post_process is True, the tree is post processed using the
        algorithm provided by Hay et al. (2009).

        :param data: data to update the tree
        :param eps: privacy parameter
        :param protocol: protocol to use for LDP frequency estimation
        :param post_process: bool, if True the tree is post processed and the cdf is computed
        """

        # run the LDP protocol
        servers, counts = ldp_protocol(data=data,
                                       eps=eps,
                                       tree=self,
                                       protocol=protocol)

        # update the attributes of the Private tree, do not update the root
        for i, level_attributes in enumerate(self.attributes):
            if i == 0:  # the root gets 1.
                self.attributes[i] = [1.]
                continue
            for j in range(len(level_attributes)):
                # as the root is not updated, we need to shift the index by 1
                self.attributes[i][j] = get_frequency(servers[i - 1], counts[i - 1], j)
        self.N = sum(counts)
        if post_process:
            self.post_process()

    def post_process(self):
        """
        Post process the tree by using the algorithm provided in the paper.

        Hay, Michael, et al. "Boosting the accuracy of differentially-private histograms through consistency." arXiv preprint arXiv:0904.0942 (2009).
        """
        B = self.b
        # Step 1: Weighted Averaging (from leaves not included to root)
        for level in reversed(range(self.depth - 1)):
            i = self.get_height(level)
            factor_1 = (B ** i - B ** (i - 1)) / (B ** i - 1)
            factor_2 = (B ** (i - 1) - 1) / (B ** i - 1)
            children_sum = sum_chunks(np.array(self.attributes[level + 1]), B)
            self.attributes[level] = [
                factor_1 * self.attributes[level][j] + factor_2 * children_sum[j]
                for j in range(len(self.attributes[level]))
            ]

        # Step 2: Mean Consistency (from root not included to leaves)
        for level in range(1, self.depth):
            parent_attributes_rep = np.repeat(self.attributes[level - 1], B)
            children_sum = np.repeat(sum_chunks(np.array(self.attributes[level]), B), B)
            self.attributes[level] = [
                self.attributes[level][j] + (1 / B) * (parent_attributes_rep[j] - children_sum[j])
                for j in range(len(self.attributes[level]))
            ]

        # The order of computation of range query is not important thanks to post processing
        self.cdf = self.attributes[-1]

    #######################
    ### QUERY FUNCTIONS ###
    #######################

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

    def get_range_query(self, left: int, right: int, normalized: bool = False) -> float:
        """
        Compute range query
        :param left: left bound of the range
        :param right: right bound of the range
        :param normalized: if True, the result is normalized by the total number of users that updated the tree

        :return: range query
        """

        assert 0 <= left <= right <= self.B, "Left and right must be between 0 and B"
        # compute right quantile
        result_right = self.cdf[right]
        # compute left quantile
        result_left = self.cdf[left]
        if normalized:
            return result_right - result_left
        else:
            return (result_right - result_left) * self.N

    def get_bins(self, quantiles: list[float], alpha: float) -> list[tuple[int, int]]:
        """
        Return a list of bins that contains quantiles q-alpha and q+alpha for each quantile q in quantiles.

        :param quantiles: list of quantiles
        :param alpha: error parameter

        :return: list of bins as tuples
        """
        assert 0 <= alpha <= 0.5, "Alpha must be between 0 and 0.5"
        assert all(0 < q < 1 for q in quantiles), "Quantiles must be between 0 and 1"

        # sort the quantiles
        quantiles = sorted(quantiles)
        bins = []
        for q in quantiles:
            # get the left and right quantile
            left = self.get_quantile(max(q - alpha, 0))
            right = self.get_quantile(min(q + alpha, 1))
            # sort left and right (they might be inverted)
            left, right = min(left, right), max(left, right)
            # append the bin
            bins.append((left, right))
        return bins

    ######################################################
    ### Function useless if the tree is post processed ###
    ######################################################

    def get_range_query_bary(self, left: int, right: int, normalized: bool = False) -> float:
        """
        Compute range query using bary indexing.
        :param left: left bound of the range
        :param right: right bound of the range
        :param normalized: if True, the result is normalized by the total number of users that updated the tree

        :return: range query
        """
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


def get_frequency(server, count, item) -> float:
    """
    Estimate the frequency of an item using the server and the count.
    :param server: a server (an instance of LDP Frequency Oracle server of pure_ldp package)
    :param count: the count of the data (server returns absolute frequency)
    :param item: the item to estimate
    """
    return server.estimate(item, suppress_warnings=True) / count


def sum_chunks(arr: np.array, chunk_size: int) -> np.array:
    """
    Sums chunks of the array.

    :param arr: Input numpy array
    :param chunk_size: Size of each chunk
    :return: Numpy array with summed chunks
    """
    # Ensure the array length is a multiple of chunk_size
    assert len(arr) % chunk_size == 0, "Array length must be a multiple of chunk size"

    # Reshape the array to have shape (-1, chunk_size)
    reshaped = arr.reshape(-1, chunk_size)
    # Sum along the second axis (axis=1)
    return reshaped.sum(axis=1)
