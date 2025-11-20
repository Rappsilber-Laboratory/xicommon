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

from libc.math cimport log

def fast_candidate_scores(np.intp_t num_candidates, np.float_t num_possible,
                          const np.float_t[:] masses, np.float_t[:, :] limits,
                          np.intp_t[:] candidate_indices):
    """
    Score candidates according to matches of mass ranges to reference masses.

    :param num_candidates: Number of candidates.
    :param num_possible: Number of possible candidates.
    :param masses: Reference masses to search within, in ascending order.
    :param limits: Array of lower and upper limits for each search mass. The shape
                   is (2, N), where N is the number of search masses. The first
                   dimension contains the lower and upper limits in that order.
    :param candidate_indices: Candidate indices associated with each range.

    :return: (ndarray) Scores for each candidate.
    """

    # Find the indices of the lowest and highest matching values in the table.
    matches = np.searchsorted(masses, limits)

    # Create array for scores initialise as 0 (=log(1) ~ totally random)
    scores_ndarray = np.zeros(num_candidates, np.float64)
    cdef np.float_t[::1] scores_view = scores_ndarray

    # Get memory views into the matches
    cdef np.intp_t[::1] start_indices = matches[0]
    cdef np.intp_t[::1] end_indices = matches[1]

    # Iterate over the matches to calculate the scores 
    cdef np.intp_t i, imax = start_indices.shape[0]
    cdef np.intp_t count
    with nogil, cython.boundscheck(False), cython.wraparound(False), \
            cython.initializedcheck(False), cython.cdivision(True):
        for i in range(imax):
            count = end_indices[i] - start_indices[i]
            # the final score used in most cases is -log(peakscore1*peakscore2*...)
            # using log(a*b*c) = log(a)+log(b)+log(c) we can transform that to
            # score = -log(peakscore1)-log(peakscore2)-...
            # If we don't use the log here we can run into cases where the score
            # gets smaller then the smallest representable float number and gets
            # turned into 0.
            # -log(0) = inf is a rather bad result for us.
            # By doing the log-conversion here - we translate the numbers into a
            # range that is unlikely to produce any overflow.
            # Additionally, we accept one "mock fragment" as base line in case there
            # is no fragment found or EmptyDatabase is used.
            scores_view[candidate_indices[i]] -= log((count + 1) / (num_possible + 1))

    return scores_ndarray
