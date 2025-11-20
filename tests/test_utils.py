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

from xicommon.utils import *
from numpy.testing import assert_array_equal


def test_boolean_group():
    bool_arr = np.array([
        False,
        False, True,
        False, False, False,
        True, True,
        False
    ])
    first_idx = np.array([0, 1, 3, 6, 8])
    last_idx = np.array([0, 2, 5, 7, 8])

    test = boolean_group(bool_arr, first_idx, last_idx)
    expected_n = np.array([0, 1, 0, 2, 0])
    assert_array_equal(expected_n, test)


def test_get_chunks():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    chunks = [x for x in get_chunks(10, 3)]

    for cid, c in enumerate(chunks):
        assert c[0] == cid * 3
        assert c[1] == min(3, 10 - cid * 3)
        assert np.array_equal(a[c[2]], a[cid * 3:cid * 3 + min(3, 10 - cid * 3)])
