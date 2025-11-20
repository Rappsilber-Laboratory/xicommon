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
from libc.string cimport memcmp

def fast_unique(array, return_index=False, return_inverse=False,
        return_counts=False, order=None, presorted=False):
    """
    Optimised equivalent to numpy.unique(axis=0), for pre-sorted input.

    :param array: (ndarray) Array to search for unique elements, ndim=1 or more. The elements
                            of axis 0 must be either single values or C-contiguous arrays.
    :param order: (ndarray, np.intp, ndim=1) Sort order of 'array'. If this is given, then
                                             'array' itself does not need to be sorted.
    :param presorted: (bool) Whether 'array' is already sorted.

    The return_index, return_inverse and return_counts parameters behave the same as for
    numpy.unique.
    """

    if order is None and not presorted:
        raise ValueError("Input must either be presorted, or an order must be provided")

    if array.ndim > 1 and array.shape[0] > 0 and not array[0].flags.c_contiguous:
        raise ValueError("Subarrays must be C-contiguous")

    base = array if array.base is None else array.base
    cdef np.uint8_t[::1] data = np.frombuffer(base, np.uint8)
    cdef np.intp_t length = array.shape[0]
    cdef size_t stride = array.strides[0]
    cdef size_t itemsize = array.itemsize
    if array.ndim > 1:
        itemsize *= np.product(array.shape[1:])
    cdef np.intp_t[::1] order_view
    diff_ndarray = np.empty(length, bool)
    cdef np.uint8_t[::1] diff_view = diff_ndarray
    cdef np.intp_t[::1] inverse_view
    cdef np.intp_t unique_index
    cdef np.intp_t i, j
    cdef np.uint8_t diff

    with cython.boundscheck(False), cython.wraparound(False):

        diff_view[0] = 1
        unique_index = 0
        if order is not None:
            order_view = order
            if return_inverse:
                inverse_ndarray = np.empty(length, np.intp)
                inverse_view = inverse_ndarray
                inverse_view[order_view[0]] = 0
                with nogil:
                    for i in range(1, length):
                        diff = memcmp(&data[stride * order_view[i]],
                                      &data[stride * order_view[i - 1]],
                                      itemsize) != 0
                        diff_view[i] = diff
                        unique_index += diff
                        inverse_view[order_view[i]] = unique_index
            else:
                with nogil:
                    for i in range(1, length):
                        j = order_view[i]
                        diff = memcmp(&data[stride * order_view[i]],
                                      &data[stride * order_view[i - 1]],
                                      itemsize) != 0
                        diff_view[i] = diff
                        unique_index += diff
        else:
            if return_inverse:
                inverse_ndarray = np.empty(length, np.intp)
                inverse_view = inverse_ndarray
                inverse_view[0] = 0
                with nogil:
                    for i in range(1, length):
                        diff = memcmp(&data[stride * i],
                                      &data[stride * (i - 1)],
                                      itemsize) != 0
                        diff_view[i] = diff
                        unique_index += diff
                        inverse_view[i] = unique_index
            else:
                with nogil:
                    for i in range(1, length):
                        diff = memcmp(&data[stride * i],
                                      &data[stride * (i - 1)],
                                      itemsize) != 0
                        diff_view[i] = diff
                        unique_index += diff

    cdef np.intp_t num_unique = unique_index + 1
    indices_ndarray = np.empty(num_unique, np.intp)
    cdef np.intp_t[::1] indices_view = indices_ndarray
    cdef np.intp_t[::1] counts_view

    with cython.boundscheck(False), cython.wraparound(False):

        if order is not None:
            indices_view[0] = order_view[0]
        else:
            indices_view[0] = 0

        unique_index = 0

        if return_counts:
            counts_ndarray = np.zeros(num_unique, np.intp)
            counts_view = counts_ndarray
            counts_view[0] = 1
            with nogil:
                if order is not None:
                    for i in range(1, length):
                        unique_index += diff_view[i]
                        indices_view[unique_index] = order_view[i]
                        counts_view[unique_index] += 1
                else:
                    for i in range(1, length):
                        unique_index += diff_view[i]
                        indices_view[unique_index] = i
                        counts_view[unique_index] += 1
        else:
            with nogil:
                if order is not None:
                    for i in range(1, length):
                        unique_index += diff_view[i]
                        indices_view[unique_index] = order_view[i]
                else:
                    for i in range(1, length):
                        unique_index += diff_view[i]
                        indices_view[unique_index] = i

    ret = (array[indices_ndarray],)

    if return_index:
        ret += (indices_ndarray,)
    if return_inverse:
        ret += (inverse_ndarray,)
    if return_counts:
        ret += (counts_ndarray,)
    return ret
