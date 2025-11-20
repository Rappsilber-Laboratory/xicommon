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

from xicommon.cython import copy_byte_ranges, argsort_byte_ranges
import numpy as np


def test_argsort_byte_ranges():

    # Source sequence strings
    src_strings = np.array([
        b'PEP',
        b'TID',
        b'KLM',
        b'LAK',
        b'KEK',
        b'LAG',
        b'LVD',
        b'LVT',
        b'AKY'])

    # Indices into the source strings for each partial sequence
    src_indices = np.array([3, 1, 2, 0, 5, 5], np.intp)

    # Start and end indices for each partial sequence
    starts = np.array([1, 0, 2, 1, 0, 0], np.intp)
    ends = np.array([3, 1, 3, 3, 3, 2], np.intp)

    # View of the source strings as a 2D byte array.
    src_bytes = src_strings.view(np.uint8).reshape(len(src_strings), -1)

    # Run in-place argsort
    order = argsort_byte_ranges(src_bytes, src_indices, starts, ends)

    # Generate the equivalent result by copying the ranges to a new array,
    # and using np.argsort on the result viewed as a string array.
    range_bytes = np.zeros((len(src_indices), src_strings.itemsize), np.uint8)
    copy_byte_ranges(src_bytes, range_bytes, src_indices, starts, ends)
    range_strings = range_bytes.reshape(-1).view(src_strings.dtype)
    expected = np.argsort(range_strings)

    # Check results match
    np.testing.assert_array_equal(order, expected)
