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
from libc.string cimport memcpy

def inplace_reorder(array, indices):
    """
    Reorder the first or only axis of 'array' into the order given by 'indices', without
    consuming additional input-sized memory. Both arrays are modified in the process: the
    indices cannot be reused afterwards!

    :param array: (ndarray) Array to be reordered, ndim=1 or more. The elements of axis 0
                            must be either single values or C-contiguous arrays.

    :param indices: (ndarray, np.intp, ndim=1) Desired order. Must be C-contiguous.
    """

    if array.ndim > 1 and array.shape[0] > 0 and not array[0].flags.c_contiguous:
        raise ValueError("Subarrays must be C-contiguous")

    base = array if array.base is None else array.base
    cdef np.uint8_t[::1] data = np.frombuffer(base, dtype=np.uint8)
    cdef np.intp_t length = array.shape[0]
    cdef size_t stride = array.strides[0]
    cdef size_t itemsize = array.itemsize
    if array.ndim > 1:
        itemsize *= np.product(array.shape[1:])
    cdef np.intp_t[::1] indices_view = indices
    cdef np.uint8_t[::1] tmp = np.empty(itemsize, np.uint8)
    cdef np.intp_t i, j, k

    with nogil, cython.boundscheck(False), cython.wraparound(False):
        for i in range(length):
            memcpy(&tmp[0], &data[stride * i], itemsize)
            j = i
            while True:
                k = indices_view[j]
                indices_view[j] = j
                if k == i:
                    break
                memcpy(&data[stride * j], &data[stride * k], itemsize)
                j = k
            memcpy(&data[stride * j], &tmp[0], itemsize)
