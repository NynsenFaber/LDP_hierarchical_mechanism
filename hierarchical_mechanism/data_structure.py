import math
from bisect import bisect
from typing import Union

import numpy as np


class Tree:
    """
    Class to represent a tree structure for the hierarchical mechanism. It works for a bounded data domain of the
    form [0, B].
    """

    def __init__(self, B: int, b: int):
        """
        Constructor

        :param B: bound of the data
        :param b: branching factor of the tree
        """
        self.B = B
        self.b = b
        self.depth = math.ceil(math.log(B, b))
        self.intervals = get_bary_partition(self.B, self.b)
        # attributes have the same shape of intervals but initialized with zeros (no root)
        self.attributes = [[0] * len(interval) for interval in self.intervals[1:]]
        print(len(self.attributes))
        self.N = None  # total number of users that updated the tree

    def find_interval_index(self, value: int, level: int) -> int:
        """
        Find the index of the subinterval where y belongs

        :param value: the value to find the interval index for
        :param level: the level of the tree to consider

        :return: the index of the subinterval where y belongs
        """
        assert 0 <= level <= self.depth, "The level must be between 0 and the depth of the tree"
        return find_interval_index(self.intervals[level], value)

    def get_bary_decomposition(self, value: Union[int, float]) -> list[list[int]]:
        """
        Compute the bary decomposition of a value

        :param value: the value to decompose

        :return: the bary decomposition of the value
        """
        return get_bary_decomposition(self.intervals, value)

    def get_bary_decomposition_index(self, value: Union[int, float]) -> list[tuple[int, int]]:
        """
        Compute the bary decomposition of a value

        :param value: the value to decompose

        :return: the bary decomposition of the value
        """
        return get_bary_decomposition_index(self.b, self.depth, value)

    def get_bary_decomposition(self, value: Union[int, float]) -> list[list[int]]:
        """
        Compute the bary decomposition of a value

        :param value: the value to decompose

        :return: the bary decomposition of the value
        """
        return get_bary_decomposition(self.intervals, value)

    # def get_quantile(self, quantile: float):
    #     value = math.ceil(quantile * self.N)
    #     indices = self.get_bary_decomposition_index(value)
    #     result = 0
    #     for i, j in indices:
    #         # attributes are normalized so we are summing frequencies
    #         result += self.attributes[i][j]
    #     return result

    def compute_cdf(self):
        cdf = np.zeros(self.B)
        for i in range(self.B):
            indices = self.get_bary_decomposition_index(i)
            result = 0
            for i, j in indices:
                # attributes are normalized so we are summing frequencies
                result += self.attributes[i][j]
            cdf[i] = result
        return cdf


def get_bary_partition(B: Union[float, int], b: int) -> list[list[list[int]]]:
    """
    Function to get the b-adic partition of the data.

    :param B: bound of the data
    :param b: branching factor of the tree

    :return: the b-adic partition of the data

    Example 1:
    B = 8
    b = 2
    get_bary_partition(B, b) -> [[[0, 8]],
                                [[0, 4], [4, 8]],
                                [[0, 2], [2, 4], [4, 6], [6, 8]],
                                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]]

    Example 2:
    B = 12
    b = 3
    get_bary_partition(B, b) -> [[[0, 27]],
                                [[0, 9], [9, 18], [18, 27]],
                                [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [15, 18], [18, 21], [21, 24], [24, 27]],
                                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], ... [26, 27]]]
    """
    # Calculate the depth of the tree based on B and b
    depth = math.ceil(math.log(B, b))
    # Initialize the results list with the root interval
    results = [[[0, b ** depth]]]
    # Iterate over each level of the tree
    for i in range(1, depth + 1):
        # Initialize the list for the current level
        inner_results = []
        # Iterate over each interval in the current level
        for j in range(b ** i):
            # Calculate the start and end of the interval
            inner_results.append([j * b ** (depth - i), (j + 1) * b ** (depth - i)])
        # Add the intervals of the current level to the results
        results.append(inner_results)
    # Return the list of intervals for all levels
    return results


def find_interval_index(intervals: list[list], value: int) -> int:
    """
    Find the index of the subinterval where y belongs

    :param intervals: a list of intervals
    :param value: the value to find the interval index for

    :return: the index of the subinterval where y belongs

    Example 1:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = 15
    find_interval_index(intervals, value) -> 1 so intervals[1] = [10, 20]

    Example 2:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = 10
    find_interval_index(intervals, value) -> 1 so intervals[0] = [10, 15]
    As the right bound is not included in the interval, the index is 1 instead of 0

    Example 3:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = 33
    find_interval_index(intervals, value) -> 2 so intervals[2] = [20, 30]
    The value is clipped

    Example 4:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = -5
    find_interval_index(intervals, value) -> 0 so intervals[0] = [0, 10]
    The value is clipped
    """
    # Extract the starting points of each interval
    starts = [interval[0] for interval in intervals]
    # Use bisect to find where `value` would fit
    # returns an insertion point which comes after (to the right of) any existing entries of value in starts
    index = bisect(starts, value)
    # index - 1 is returned as the index are [j B^i, (j+1)B^i), so the right bound is not included
    return index - 1 if index > 0 else 0


def get_bary_representation(value: int, b: int, length: int) -> list[int]:
    """
    Compute the bary representation of a value, for b=2 is the binary representation

    :param value: the value to represent
    :param b: the base of the representation
    :param length: the length of the representation

    :return: the bary representation of the value
    """
    # raise an error if the length is not enough to represent the value
    assert value < b ** length, "The value cannot be represented with the given length"

    # Initialize the representation
    representation = [0] * length
    # Iterate over the length of the representation
    for i in range(length - 1, -1, -1):
        # Compute the value of the current digit
        representation[i] = value % b
        # Update the value for the next iteration
        value = value // b
    return representation


def get_bary_decomposition_index(b: int,
                                 length: int,
                                 value: Union[int, float]) -> list[tuple[int, int]]:
    # Apply the floor and add one, this is because the bounds are [left, right)
    # consider value = 4 for example, then
    # we search for 5 so to return [[0, 4], [4,5]]
    value = math.floor(value) + 1
    # If the value exceeds the maximum representable value, return [(0, 0)]
    if value >= b ** (length - 1):
        return [(1, x) for x in range(b)]

    results = []
    # Get the bary representation of the value
    bary_rep = get_bary_representation(value, b, length)
    offset = 0
    # Iterate over each level of the representation
    for i in range(1, length):
        # Calculate the index for the current level
        # index = bary_rep[i] + (1 if i == length - 1 else 0)
        index = bary_rep[i]
        # Extend the results with the current level indices
        results.extend((i, j) for j in range(offset, offset + index))
        # Update the offset for the next level
        offset = offset * b + index * b
    return results


def get_bary_decomposition(bary_partition: list[list[list[int]]],
                           value: Union[int, float]) -> list[list[int]]:
    # Get the indices of the bary decomposition
    indices = get_bary_decomposition_index(len(bary_partition[1]), len(bary_partition), value)
    # Use list comprehension for faster results
    return [bary_partition[i][j] for i, j in indices]


def test_get_bary_partition():
    B = 8
    b = 2
    assert get_bary_partition(B, b) == [[[0, 8]],
                                        [[0, 4], [4, 8]],
                                        [[0, 2], [2, 4], [4, 6], [6, 8]],
                                        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]]

    B = 12
    b = 3
    assert get_bary_partition(B, b) == [[[0, 27]],
                                        [[0, 9], [9, 18], [18, 27]],
                                        [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [15, 18], [18, 21], [21, 24],
                                         [24, 27]],
                                        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                                         [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],
                                         [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25],
                                         [25, 26], [26, 27]]]


def test_find_interval_index():
    intervals = [[0, 10], [10, 20], [20, 30]]
    assert find_interval_index(intervals, 15) == 1
    assert find_interval_index(intervals, 10) == 1
    assert find_interval_index(intervals, 33) == 2
    assert find_interval_index(intervals, -5) == 0


def test_tree():
    B = 8
    b = 2
    tree = Tree(B, b)

    assert tree.find_interval_index(15, 1) == 1
    assert tree.find_interval_index(10, 1) == 1
    assert tree.find_interval_index(33, 1) == 1
    assert tree.find_interval_index(-5, 1) == 0
    assert tree.find_interval_index(7, 3) == 7

    assert tree.get_bary_decomposition(7) == [[0, 4], [4, 8]]


test_get_bary_partition()
test_find_interval_index()
test_tree()

