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

from xicommon.cython import copy_byte_ranges
import numpy as np


def test_copy_byte_ranges_uint8():

    src = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]], np.uint8)

    src_indices = np.array([0, 0, 1, 2], np.intp)
    starts = np.array([0, 1, 2, 3], np.intp)
    ends = np.array([4, 5, 3, 5], np.intp)

    expected = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [8, 0, 0, 0],
        [14, 15, 0, 0]], np.uint8)

    dest = np.zeros((4, 4), np.uint8)
    copy_byte_ranges(src, dest, src_indices, starts, ends)

    np.testing.assert_array_equal(dest, expected)


def test_copy_byte_ranges_string():

    src_strings = np.array([
        b'BUMPS',
        b'ARG',
        b'TEST'])

    src_indices = np.array([0, 0, 1, 2], np.intp)
    starts = np.array([0, 1, 2, 1], np.intp)
    ends = np.array([4, 5, 3, 3], np.intp)

    expected_strings = np.array([
        b'BUMP',
        b'UMPS',
        b'G',
        b'ES'])

    src = src_strings.view(np.uint8).reshape(3, 5)
    dest = np.zeros((4, 4), np.uint8)
    copy_byte_ranges(src, dest, src_indices, starts, ends)

    dest_strings = dest.reshape(16).view('S4')
    np.testing.assert_array_equal(dest_strings, expected_strings)
