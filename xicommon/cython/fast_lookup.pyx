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

from xicommon.utils import get_chunk_indices
import numpy as np
cimport numpy as np
cimport cython

# Datatype for results
result_dtype = np.dtype([('mass_index', np.intp), ('id', np.intp)])

def fast_lookup(const np.float_t[:] masses, ids, np.float_t[:, :, :] limits,
                bint return_matches, bint return_unique, bint return_counts):
    """
    Look up matching IDs for given masses.

    :param masses: Reference masses to search within, in ascending order.
    :param ids: Array of IDs for the molecules corresponding to the reference masses.
                If given, the 'id' field in the result will refer to the IDs from
                this array.
    :param limits: Array of lower and upper limits for each search mass. The shape
                   is (2, N, M), where N is the number of search masses and M is
                   the number of alternative values for each (e.g. delta masses).
                   The first dimension contains the lower and upper limits in that
                   order.
    :param return_matches: If True, returns a structured array with 'mass_index' and
                          'id' fields, The 'mass_index' field will contain the indices
                          from the N dimension of 'limits' for each match.
                          If the 'ids' array is provided, the 'id' field will contain
                          the IDs of the molecules corresponding to the reference mass
                          for each match. Otherwise, the 'id' field will simply contain
                          the matching indices of the reference masses.
    :param return_unique: If True, also return an array with the unique indices of
                          the search masses that matched.
    :param return_counts: If True, also return an array with the number of matches
                          for each unique index of the search masses that matched.
    :return: As above, depending on parameters.
    """
    cdef bint have_ids = (ids is not None)

    # Find the indices of the lowest and highest matching values in the table.
    matches = np.searchsorted(masses, limits)

    # Get memory views into the matches
    cdef np.intp_t[:, ::1] start_indices = matches[0]
    cdef np.intp_t[:, ::1] end_indices = matches[1]

    # Iterate once over the matches to get the total number of matches and
    # the number of unique matching mass indices.
    cdef np.intp_t total_matches = 0
    cdef np.intp_t num_matching_masses = 0
    cdef np.intp_t last_matching_mass_index = -1
    cdef np.intp_t i, imax = start_indices.shape[0]
    cdef np.intp_t j, jmax = start_indices.shape[1]
    cdef np.intp_t count
    with nogil, cython.boundscheck(False), cython.wraparound(False):
        if return_unique or return_counts:
            for i in range(imax):
                for j in range(jmax):
                    count = (end_indices[i, j] - start_indices[i, j])
                    total_matches += count
                    if count > 0 and i != last_matching_mass_index:
                        last_matching_mass_index = i
                        num_matching_masses += 1
        elif return_matches:
            for i in range(imax):
                for j in range(jmax):
                    count = (end_indices[i, j] - start_indices[i, j])
                    total_matches += count

    if return_matches:
        # Allocate result table now that we know the size required
        result = np.empty(total_matches, result_dtype)

    # Allocate supplemental arrays now that we know how many masses matched
    cdef np.intp_t[::1] unique_indices_view
    cdef np.intp_t[::1] unique_counts_view
    if return_unique:
        unique_indices_ndarray = np.empty(num_matching_masses, np.intp)
        unique_indices_view = unique_indices_ndarray
    if return_counts:
        unique_counts_ndarray = np.empty(num_matching_masses, np.intp)
        unique_counts_view = unique_counts_ndarray

    # Get memory views into the result columns and IDs if provided.
    cdef np.intp_t[:] result_mass_indices
    cdef np.intp_t[:] result_ids
    cdef const np.intp_t[:] ids_view
    if return_matches:
        result_mass_indices = result['mass_index']
        result_ids = result['id']
    if ids is not None:
        ids_view = ids

    # Now iterate over the matches again to populate the result table
    cdef np.intp_t row = 0
    cdef np.intp_t index
    cdef np.intp_t unique_entry = -1
    last_matching_mass_index = -1
    with nogil, cython.boundscheck(False), cython.wraparound(False), cython.initializedcheck(False):
        if return_unique or return_counts:
            if have_ids:
                for i in range(imax):
                    for j in range(jmax):
                        for index in range(start_indices[i, j], end_indices[i, j]):
                            if return_matches:
                                result_mass_indices[row] = i
                                result_ids[row] = ids_view[index]
                                row += 1
                            if i != last_matching_mass_index:
                                last_matching_mass_index = i
                                unique_entry += 1
                                if return_unique:
                                    unique_indices_view[unique_entry] = i
                                if return_counts:
                                    unique_counts_view[unique_entry] = 0
                            if return_counts:
                                unique_counts_view[unique_entry] += 1
            else:
                for i in range(imax):
                    for j in range(jmax):
                        for index in range(start_indices[i, j], end_indices[i, j]):
                            if return_matches:
                                result_mass_indices[row] = i
                                result_ids[row] = index
                                row += 1
                            if i != last_matching_mass_index:
                                last_matching_mass_index = i
                                unique_entry += 1
                                if return_unique:
                                    unique_indices_view[unique_entry] = i
                                if return_counts:
                                    unique_counts_view[unique_entry] = 0
                            if return_counts:
                                unique_counts_view[unique_entry] += 1
        elif return_matches:
            if have_ids:
                for i in range(imax):
                    for j in range(jmax):
                        for index in range(start_indices[i, j], end_indices[i, j]):
                            result_mass_indices[row] = i
                            result_ids[row] = ids_view[index]
                            row += 1
            else:
                for i in range(imax):
                    for j in range(jmax):
                        for index in range(start_indices[i, j], end_indices[i, j]):
                            result_mass_indices[row] = i
                            result_ids[row] = index
                            row += 1

    results = tuple()

    if return_matches:
        results += (result,)
    if return_unique:
        results += (unique_indices_ndarray,)
    if return_counts:
        results += (unique_counts_ndarray,)

    if len(results) == 1:
        return results[0]
    else:
        return results
