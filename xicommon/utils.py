# Copyright (C) 2025  Technische Universitaet Berlin
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA

"""Module containing utility functions."""
import numpy as np


def get_chunks(length, max_chunk_size):
    """
    Given a length and a maximum chunk size, generate a set of chunks of appropriate size.

    For each chunk, yields a tuple of:
        - The starting index of the chunk
        - The size of the chunk
        - A slice object to retrieve this chunk
    """
    chunk_start = 0
    while length > max_chunk_size:
        yield chunk_start, max_chunk_size, slice(chunk_start, chunk_start + max_chunk_size)
        chunk_start += max_chunk_size
        length -= max_chunk_size
    yield chunk_start, length, slice(chunk_start, chunk_start + length)


# Nifty trick adapted from:
# https://stackoverflow.com/questions/22326882
#
# Given a set of index ranges described by start and end indices, and the
# lengths of each range (equal to `ends - starts`) return a single array
# containing all the indices within these ranges in order. The lengths are
# given as a parameter to save recalculating them, because when we call this,
# we have already done that. There must be no zero-length ranges in the input.

def get_chunk_indices(starts, ends, lengths):
    # Handle the zero-length case
    if len(lengths) == 0:
        return np.array([], np.intp)
    # Take the cumulative sum of the range lengths. This gives us the
    # boundaries where we will switch from one range to the next in our result.
    boundaries = np.cumsum(lengths)
    # The last boundary is equal to the total number of indices we will return.
    total_indices = boundaries[-1]
    # Now we start computing the steps between indices in our result.
    # Initialise with ones, because that will be the step between all indices
    # within the same range.
    steps = np.ones(total_indices, np.intp)
    # At the boundaries between ranges, the step is the difference between the
    # end of one range and the start of the next range.
    steps[boundaries[:-1]] = starts[1:] - ends[:-1] + 1
    # The first step needs to take us to the start of the first range.
    steps[0] = starts[0]
    # Having worked out all the steps between indices, the indices themselves
    # can now be obtained by taking the cumulative sum along the steps.
    indices = np.cumsum(steps)
    return indices


def concatenated_ranges(lengths):
    """
    Create an array that consecutively counts up to each number in lengths.

    # Another nifty trick adapted from https://stackoverflow.com/a/45126708
    :param lengths: array with numbers that should be counted up to
    :return: the equivalent of np.concatenate([np.arange(length) for length in lengths])
    """
    cumsum = np.zeros(len(lengths), dtype=lengths.dtype)
    cumsum[1:] = np.cumsum(lengths[:-1])
    cumsum = np.repeat(cumsum, lengths)
    return np.arange(cumsum.shape[0]) - cumsum


def boolean_group(bool_arr, first_idx, last_idx):
    """
    Return the number of True values in a boolean array grouped by indices.

    :param bool_arr: (ndarray) boolean array
    :param first_idx: (ndarray) array containing the first index of each group
    :param last_idx: (ndarray) array containing the last index of each group
    :return: array containing the number of True entries for each group,
    :rtype: ndarray int
    """
    cumsum = np.empty(bool_arr.size + 1)
    cumsum[0] = 0
    cumsum[1:] = np.cumsum(bool_arr)
    # how many entries per group have a True value
    n_true = cumsum[last_idx + 1] - cumsum[first_idx]
    return n_true
