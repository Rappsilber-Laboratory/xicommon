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

#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from .argsort_byte_ranges import argsort_byte_ranges
from .compare_byte_ranges cimport compare_byte_ranges, byte_ranges_context
from libc.string cimport memcpy

def unique_byte_ranges(const np.uint8_t[:, :] src, const np.intp_t[:] src_indices,
                       const np.intp_t[:] starts, const np.intp_t[:] ends):
    """
    Find unique sequences of bytes, given source ranges from a 2D byte array.

    :param src: Source array (uint8, ndim=2)
    :param src_indices: The row indices of 'src' from which to take each sequence.
    :param starts: Starting column indices of 'src' from which to take each sequence.
    :param ends: Ending column indices of 'src' from which to take each sequence.

    :return: Unique sequences, and indices into this array for each input sequence.
    """

    # Generate ordering of the ranges first
    order = argsort_byte_ranges(src, src_indices, starts, ends)

    # Prepare context structure
    cdef byte_ranges_context ctx = byte_ranges_context(src, src_indices, starts, ends)

    # Memory views and working variables
    cdef np.intp_t length = src_indices.shape[0]
    cdef np.intp_t[::1] order_view = order
    diff_ndarray = np.empty(length, bool)
    cdef np.uint8_t[::1] diff_view = diff_ndarray
    inverse_ndarray = np.empty(length, np.intp)
    cdef np.intp_t[::1] inverse_view = inverse_ndarray
    cdef np.intp_t unique_index
    cdef np.intp_t i
    cdef np.uint8_t diff

    with cython.boundscheck(False), cython.wraparound(False):

        # Iterate through the ranges in the sorted order, counting unique entries
        # and recording whether adjacent elements differ.
        diff_view[0] = 1
        unique_index = 0
        inverse_view[order_view[0]] = 0
        for i in range(1, length):
            diff = compare_byte_ranges(&order_view[i], &order_view[i - 1], <void *> ctx) != 0
            diff_view[i] = diff
            unique_index += diff
            inverse_view[order_view[i]] = unique_index

    # Allocate output array now that we know the number of unique entries
    cdef np.intp_t num_unique = unique_index + 1
    unique_ndarray = np.zeros((num_unique, src.shape[1]), np.uint8)
    cdef np.uint8_t[:, ::1] unique_view = unique_ndarray

    with nogil, cython.boundscheck(False), cython.wraparound(False):

        # The first entry in the output is the first entry in the input
        memcpy(&unique_view[0, 0],
               &src[src_indices[order_view[0]], starts[order_view[0]]],
               ends[order_view[0]] - starts[order_view[0]])
        unique_index = 0

        # Iterate through the ranges again, populating unique entries in the output
        for i in range(1, length):
            unique_index += diff_view[i]
            memcpy(&unique_view[unique_index, 0],
                   &src[src_indices[order_view[i]], starts[order_view[i]]],
                   ends[order_view[i]] - starts[order_view[i]])

    return unique_ndarray, inverse_ndarray
