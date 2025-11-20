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
from .compare_byte_ranges cimport byte_ranges_context, compare_byte_ranges

# Declare the qsort_r function (it's not in the cython-supplied imports)
cdef extern from "stdlib.h":
    void qsort_r(void *base, int nmemb, size_t size,
            int (*compar)(const void *, const void *, void *), void *arg)

def argsort_byte_ranges(const np.uint8_t[:, :] src, const np.intp_t[:] src_indices,
                        const np.intp_t[:] starts, const np.intp_t[:] ends):
    """
    Find order for sequences of bytes, given source ranges from a 2D byte array.

    :param src: Source array (uint8, ndim=2)
    :param src_indices: The row indices of 'src' from which to take each sequence.
    :param starts: Starting column indices of 'src' from which to take each sequence.
    :param ends: Ending column indices of 'src' from which to take each sequence.

    :return: Sort order for the sequences referred to.
    """
    # Prepare array of indices
    cdef np.intp_t num_inputs = src_indices.shape[0]
    order_ndarray = np.arange(num_inputs, dtype=np.intp)
    cdef np.intp_t[:] order = order_ndarray

    # Prepare context structure
    cdef byte_ranges_context ctx = byte_ranges_context(src, src_indices, starts, ends)

    # Run quicksort
    qsort_r(&order[0], num_inputs, sizeof(np.intp_t), compare_byte_ranges, <void *> ctx)

    # Return sorted indices
    return order_ndarray
