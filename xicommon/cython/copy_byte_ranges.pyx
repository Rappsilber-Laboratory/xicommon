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


def copy_byte_ranges(np.ndarray src, np.ndarray dest,
                     const np.intp_t[:] src_indices, const np.intp_t[:] starts,
                     const np.intp_t[:] ends):
    """
    Copy selected byte ranges from one 2D byte array to another.

    :param src: Source array (uint8, ndim=2)
    :param dest: Destination array (uint8, ndim=2)
    :param src_indices: The row indices of 'src' from which to copy, for each row of 'dest'.
    :param starts: Starting column indices of 'src' from which to copy, for each row of 'dest'.
    :param ends: Ending column indices of 'src' from which to copy, for each row of 'dest'.

    The copied ranges are "left aligned" in dest, i.e. each is copied starting at column 0.
    """

    for array in (src, dest):
        if array.shape[0] > 0 and not array[0].flags.c_contiguous:
            raise ValueError("Subarrays must be C-contiguous")

    cdef const np.uint8_t[:, :] src_view = src
    cdef np.uint8_t[:, :] dest_view = dest

    cdef np.intp_t i, j
    cdef np.intp_t length

    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(len(starts)):
            length = ends[i] - starts[i]
            memcpy(&dest_view[i, 0], &src_view[src_indices[i], starts[i]], length)
