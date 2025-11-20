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

cimport numpy as np
from libc.string cimport memcmp
cimport cython

@cython.final
cdef class byte_ranges_context:

    def __init__(self, const np.uint8_t[:, :] src,
                 const np.intp_t[:] src_indices,
                 const np.intp_t[:] starts,
                 const np.intp_t[:] ends):
        self.src = src
        self.src_indices = src_indices
        self.starts = starts
        self.ends = ends

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int compare_byte_ranges(const void *a, const void *b, void *arg) noexcept:
    """
    Compare helper function. Given pointers to two entries in the order array, return:
    - negative if the sequence referred to by a precedes that referred to by b.
    - positive if the sequence referred to by b precedes that referred to by a.
    - zero if the sequences are equal.
    """
    # Dereference parameters
    cdef np.intp_t ai = (<np.intp_t *> a)[0]
    cdef np.intp_t bi = (<np.intp_t *> b)[0]
    cdef byte_ranges_context ctx = <object> arg

    # Get lengths of each referred sequence
    cdef np.intp_t a_len = ctx.ends[ai] - ctx.starts[ai]
    cdef np.intp_t b_len = ctx.ends[bi] - ctx.starts[bi]

    # Compare the bytes of the two sequences, up to their minimum common length
    cdef bint memcmp_result = memcmp(&ctx.src[ctx.src_indices[ai], ctx.starts[ai]],
                                     &ctx.src[ctx.src_indices[bi], ctx.starts[bi]],
                                     min(a_len, b_len))
    if memcmp_result == 0:
        # One sequence is a prefix of the other - the shorter one comes first
        return a_len - b_len
    else:
        # The sequences differ - let the memcmp result order them
        return memcmp_result
