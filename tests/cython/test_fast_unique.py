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

from xicommon.cython import fast_unique
import numpy as np
import itertools


def test_fast_unique():

    for test_array in (
            np.array([3, 2, 1, 3, 5, 1]),
            np.array([b'QUUX', b'FOO', b'QUUX', b'FOO', b'BAR', 'BAZ']),
            np.array([
                [3, 2, 1],
                [0, 0, 1],
                [3, 2, 1],
                [0, 0, 1],
                [1, 2, 3]]),
            np.array([
                (True, 1, b'FOO'),
                (True, 2, b'FOO'),
                (True, 2, b'BAR'),
                (True, 1, b'FOO'),
                (False, 1, b'QUUX'),
                (False, 1, b'QUUX'),
                (False, 1, b'QUU')],
                dtype=[
                    ('bool', bool),
                    ('int', int),
                    ('string', 'S4')])):

        if test_array.ndim == 1:
            test_order = np.argsort(test_array, axis=0)
        else:
            test_order = np.lexsort(test_array.T[::-1])

        expected_unique, expected_counts = np.unique(test_array, axis=0, return_counts=True)

        for return_index, return_inverse, return_counts in \
                itertools.product((True, False), repeat=3):

            for array, presorted, order in (
                    (test_array, False, test_order),
                    (test_array[test_order], True, None)):

                result = fast_unique(array,
                                     presorted=presorted,
                                     order=order,
                                     return_index=return_index,
                                     return_inverse=return_inverse,
                                     return_counts=return_counts)

                unique = result[0]
                np.testing.assert_array_equal(unique, expected_unique)

                extras = list(result[1:])

                if return_counts:
                    counts = extras.pop()
                    np.testing.assert_array_equal(counts, expected_counts)

                if return_inverse:
                    inverse = extras.pop()
                    np.testing.assert_array_equal(unique[inverse], array)

                if return_index:
                    indices = extras.pop()
                    np.testing.assert_array_equal(array[indices], unique)
