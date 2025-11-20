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


def num_lexico_permutation(const np.uint8_t[:, :] src):
    """
    Calculate the number of lexicographical permutations of an array of uppercase char bytes.

    :param src: Source array (uint8, ndim=2)
    :return: Number of lexicographical permutations
    """
    result = np.empty(len(src), np.float64)
    cdef np.float64_t[:] result_c = result

    cdef np.intp_t i, imax = src.shape[0]
    cdef np.intp_t j, jmax = src.shape[1]
    cdef np.intp_t total_char_count
    # define an array to count uppercase characters 65-90
    char_counts = np.zeros(26, dtype=np.intp)
    cdef np.intp_t[:] char_counts_c = char_counts
    cdef np.intp_t k, kmax = 26
    cdef np.float64_t divisor

    factorials = np.empty(src.shape[1], dtype=np.float64)
    cdef np.float64_t[:] factorials_c = factorials
    factorials_c[0] = 1

    with cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
        # precalculate factorials up to max length
        for j in range(1, jmax):
            factorials_c[j] = factorials_c[j-1] * (j+1)
        # loop over rows
        for i in range(imax):
            # reset counters
            total_char_count = 0
            for c in range(26):
                char_counts_c[c] = 0
            # loop over characters
            for j in range(jmax):
                # stop if we encounter a zero (end of sequence)
                if src[i, j] == 0:
                    break
                total_char_count += 1
                char_counts_c[src[i, j]-65] += 1
            # loop over char_counts
            divisor = 1
            for k in range(kmax):
                if char_counts_c[k] != 0:
                    divisor *= factorials_c[char_counts_c[k]-1]
            result_c[i] = factorials_c[total_char_count-1] / divisor
        return result