import math
from bisect import bisect
from typing import Union

import numpy as np


class TreeBary:
    """
    Class to represent a tree structure for the hierarchical mechanism. It works for a bounded data domain of the
    form [0, B].
    """

    def __init__(self, B: int, b: int, set_intervals: bool = True):
        """
        Constructor

        :param B: bound of the data, of the form [0, B]
        :param b: branching factor of the tree
        :param set_intervals: whether to set the intervals of the tree or not. If False, the intervals are not set
            and the data structure is more space efficient.
        """
        self.b = b
        # e.g. for B=8, b=2, depth=3, while for B=9, b=2, depth=4 as 2^4=16>=9
        self.depth = math.ceil(math.log(B, b)) + 1
        # max value of the data domain
        self.B = b ** (self.depth - 1)
        if set_intervals:
            # get the b-ary partition of the data, shape: tree level -> interval index -> interval
            self.intervals: list[list[list[int]]] = get_bary_partition(self.B, self.b)
        else:
            self.intervals = None

    def find_interval_index(self, value: int, level: int) -> int:
        """
        Find the index of the subinterval where y belongs

        :param value: the value to find the interval index for
        :param level: the level of the tree to consider

        :return: the index of the subinterval where y belongs
        """
        assert 0 <= level < self.depth, "The level must be between 0 and the depth of the tree"
        if self.intervals is None:
            return find_interval_index(level, self.b, self.depth, value)
        else:
            return find_interval_index_with_intervals(self.intervals[level], value)

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

    def get_level_indices(self, level: int) -> list[int]:
        """
        Get the indices of the subintervals at a given level

        :param level: the level of the tree

        :return: the indices of the subintervals at the given level
        """
        assert 0 <= level < self.depth, "The level must be between 0 and the depth of the tree"
        return range(self.b ** level)

    def get_height(self, level) -> int:
        """
        Given a level (0 for the root, self.depth-1 for the leaves), return the height of the tree (1 for the leaves,
        self.depth for the root)

        :return: the height of the tree
        """
        assert 0 <= level < self.depth, "The level must be between 0 and the depth of the tree"
        return self.depth - level

    def update_bary_partition(self):
        self.intervals = get_bary_partition(self.B, self.b)


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


def find_interval_index_with_intervals(intervals: list[list], value: int) -> int:
    """
    Find the index of the subinterval where y belongs

    :param intervals: a list of intervals
    :param value: the value to find the interval index for

    :return: the index of the subinterval where y belongs

    Example 1:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = 15
    find_interval_index_with_intervals(intervals, value) -> 1 so intervals[1] = [10, 20]

    Example 2:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = 10
    find_interval_index_with_intervals(intervals, value) -> 1 so intervals[0] = [10, 15]
    As the right bound is not included in the interval, the index is 1 instead of 0

    Example 3:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = 33
    find_interval_index_with_intervals(intervals, value) -> 2 so intervals[2] = [20, 30]
    The value is clipped

    Example 4:
    intervals = [[0, 10], [10, 20], [20, 30]]
    value = -5
    find_interval_index_with_intervals(intervals, value) -> 0 so intervals[0] = [0, 10]
    The value is clipped
    """
    # Extract the starting points of each interval
    starts = [interval[0] for interval in intervals]
    # Use bisect to find where `value` would fit
    # returns an insertion point which comes after (to the right of) any existing entries of value in starts
    index = bisect(starts, value)
    # index - 1 is returned as the index are [j B^i, (j+1)B^i), so the right bound is not included
    return index - 1 if index > 0 else 0


def find_interval_index(level: int, b: int, depth: int, value: int) -> int:
    """
    Space efficient version of find_interval_index_with_intervals

    Find the index of the subinterval where y belongs

    :param level: the level of the tree (0 for the root, depth-1 for the leaves)
    :param b: the base of the representation
    :param depth: the depth of the tree
    :param value: the value to find the interval index for

    :return: the index of the subinterval where y belongs

    Example 1:
    level = 1
    b = 2
    depth = 4
    value = 7
    find_interval_index(height, b, depth, value) -> 0 so intervals[0] = [0, 8]

    Example 2:
    level = 1
    b = 10
    depth = 4
    value = 20
    find_interval_index(height, b, depth, value) -> 2 so intervals[2] = [20, 30]
    As the right bound is not included in the interval, the index is 1 instead of 0

    Example 3:
    level = 2
    b = 10
    depth = 2
    value = 150
    find_interval_index(height, b, depth, value) -> 1 so intervals[1] = [100, 150]
    """
    height = depth - level
    if value <= 0:
        return 0
    if value >= b ** (depth - 1):
        return int(b ** level - 1)

    return int(np.floor(value / (b ** (height - 1))) + 1) - 1


def get_bary_representation(value: int, b: int, length: int) -> list[int]:
    """
    Compute the bary representation of a value, for b=2 is the binary representation

    :param value: the value to represent
    :param b: the base of the representation
    :param length: the length of the representation

    :return: the bary representation of the value

    Example 1:
    value = 5
    b = 2
    length = 3
    get_bary_representation(value, b, length) -> [1, 0, 1]

    Example 2:
    value = 8
    b = 3
    length = 4
    get_bary_representation(value, b, length) -> [0, 0, 2, 2]
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
    """
    It gives the indices of the tree (of the form [(level, index), (level, index), ...]) that represent the bary
    decomposition of the value.
    Note that the root is not used here.

    :param b: branching factor of the tree
    :param length: length of the bary representation
    :param value: value to decompose

    :return: list of tuples of the form [(level, index), (level, index), ...]

    Example 1:
    b = 2
    length = 3
    value = 5
    get_bary_decomposition_index(b, length, value) -> [(1, 0), (2, 2)]
    so that the bary decomposition is [[0, 4], [4, 6]]

    Example 2:
    b = 3
    length = 4
    value = 21
    get_bary_decomposition_index(b, length, value) -> [(1, 0), (1, 1), (2, 6), (3, 21)]
    so that the bary decomposition is [[0, 9], [9, 18], [18, 21], [21, 22]]
    """
    # print("length", length)
    # Apply the floor and add one, this is because the bounds are [left, right)
    # consider value = 4 for example, then we search for 5 so to return [[0, 4], [4,5]]
    value = math.floor(value) + 1
    # If the value exceeds the maximum representable value, return all the first level indices.
    if value >= b ** (length - 1):
        return [(1, x) for x in range(b)]
    elif value <= 0:
        return [(length - 1, 0)]

    results = []
    # Get the bary representation of the value
    bary_rep = get_bary_representation(value, b, length)
    # the offset is used to navigate the tree
    offset = 0
    # Iterate over each level of the representation
    for i in range(1, length):
        # Calculate the index for the current level
        index = bary_rep[i]
        # Extend the results with the current level indices
        results.extend((i, j) for j in range(offset, offset + index))
        # Update the offset for the next level
        offset = offset * b + index * b
    return results


def get_bary_decomposition(bary_partition: list[list[list[int]]],
                           value: Union[int, float]) -> list[list[int]]:
    """
    Transform the bary decomposition indices into the actual intervals

    :param bary_partition: bary partition of the data
    :param value: value to decompose

    :return: a list of intervals that represent the bary decomposition of the value
    """
    # Get the indices of the bary decomposition
    indices = get_bary_decomposition_index(len(bary_partition[1]), len(bary_partition), value)
    # Use list comprehension for faster results
    return [bary_partition[i][j] for i, j in indices]


############################################################################################################
# Test the functions
############################################################################################################


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

    B = 10
    b = 5
    final = [[x, y] for x, y in zip(range(0, 25, 1), range(1, 26, 1))]
    assert get_bary_partition(B, b) == [[[0, 25]],
                                        [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25]],
                                        final]

    B = 100
    b = 5
    second_last = [[x, y] for x, y in zip(range(0, 125, 5), range(5, 126, 5))]
    final = [[x, y] for x, y in zip(range(0, 125, 1), range(1, 126, 1))]
    assert get_bary_partition(B, b) == [[[0, 125]],
                                        [[0, 25], [25, 50], [50, 75], [75, 100], [100, 125]],
                                        second_last,
                                        final]

    B = 200
    b = 6
    second_last = [[x, y] for x, y in zip(range(0, 216, 6), range(6, 222, 6))]
    final = [[x, y] for x, y in zip(range(0, 216, 1), range(1, 222, 1))]
    assert get_bary_partition(B, b) == [[[0, 216]],
                                        [[0, 36], [36, 72], [72, 108], [108, 144], [144, 180], [180, 216]],
                                        second_last,
                                        final]


def test_find_interval_index():
    intervals = [[0, 10], [10, 20], [20, 30]]
    assert find_interval_index_with_intervals(intervals, 15) == 1
    assert find_interval_index_with_intervals(intervals, 10) == 1
    assert find_interval_index_with_intervals(intervals, 33) == 2
    assert find_interval_index_with_intervals(intervals, -5) == 0

    intervals = [[0, 25], [25, 50], [50, 75], [75, 100], [100, 125]]
    assert find_interval_index_with_intervals(intervals, 15) == 0
    assert find_interval_index_with_intervals(intervals, 25) == 1
    assert find_interval_index_with_intervals(intervals, 33) == 1
    assert find_interval_index_with_intervals(intervals, 100) == 4
    assert find_interval_index_with_intervals(intervals, 125) == 4
    assert find_interval_index_with_intervals(intervals, 126) == 4
    assert find_interval_index_with_intervals(intervals, -5) == 0


def test_bary_representation():
    assert get_bary_representation(5, 2, 3) == [1, 0, 1]
    assert get_bary_representation(8, 3, 4) == [0, 0, 2, 2]
    assert get_bary_representation(4, 3, 3) == [0, 1, 1]


def test_get_bary_decomposition_index():
    b = 2
    length = 4
    assert get_bary_decomposition_index(b, length, 5) == [(1, 0), (2, 2)]

    b = 3
    length = 4
    assert get_bary_decomposition_index(b, length, 21) == [(1, 0), (1, 1), (2, 6), (3, 21)]

    b = 5
    length = 4
    assert get_bary_decomposition_index(b, length, 27) == [(1, 0), (3, 25), (3, 26), (3, 27)]


def test_tree():
    B = 8
    b = 2
    tree = TreeBary(B, b)
    assert tree.find_interval_index(15, 1) == 1
    assert tree.find_interval_index(10, 1) == 1
    assert tree.find_interval_index(33, 1) == 1
    assert tree.find_interval_index(-5, 1) == 0
    assert tree.find_interval_index(7, 3) == 7
    assert tree.get_bary_decomposition(7) == [[0, 4], [4, 8]]
    assert tree.get_bary_decomposition_index(5) == [(1, 0), (2, 2)]

    B = 21
    b = 3
    tree = TreeBary(B, b)
    assert tree.get_bary_decomposition(21) == [[0, 9], [9, 18], [18, 21], [21, 22]]
    assert tree.get_bary_decomposition(29) == [[0, 9], [9, 18], [18, 27]]
    assert tree.get_bary_decomposition(27) == [[0, 9], [9, 18], [18, 27]]
    assert tree.get_bary_decomposition(26) == [[0, 9], [9, 18], [18, 27]]
    assert tree.get_bary_decomposition(25) == [[0, 9], [9, 18], [18, 21], [21, 24], [24, 25], [25, 26]]
    assert tree.get_bary_decomposition(0) == [[0, 1]]
    assert tree.get_bary_decomposition(-1) == [[0, 1]]

    # Space efficient version
    B = 21
    b = 3
    tree = TreeBary(B, b, set_intervals=False)
    assert tree.find_interval_index(15, 1) == 1
    assert tree.find_interval_index(10, 1) == 1
    assert tree.find_interval_index(9, 1) == 1
    assert tree.find_interval_index(33, 1) == 2
    assert tree.find_interval_index(-5, 1) == 0
    assert tree.find_interval_index(7, 3) == 7


test_get_bary_partition()
test_find_interval_index()
test_bary_representation()
test_get_bary_decomposition_index()
test_tree()
