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

from xicommon.cython import num_lexico_permutation
import numpy as np


def test_num_lexico_permutation():
    array = np.array([b'A', b'AB', b'AA', b'BAC', b'ABCD', b'DCADA'])
    array = array.view(np.uint8).reshape(len(array), -1)
    num_permutations = num_lexico_permutation(array)
    expected = np.array([
        1,  # A: 1!
        2,  # AB: 2!
        1,  # AA: 2!/2!
        6,  # BAC: 3!
        24,  # ABCD: 4!
        30,  # DCADA: 5!/(2!*2!)
    ])
    np.testing.assert_array_equal(num_permutations, expected)
