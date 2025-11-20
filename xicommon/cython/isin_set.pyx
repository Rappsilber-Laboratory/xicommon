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

"""
This module provides a set of function that can replace np.isin with an more efficient
implementation. Currently single one-dimensional vectors/single columns can be compared.

The isin functions here convert the second argument into a c++ unordered list and test the elements 
of the first argument against the second. The only requirement is that both are numpy arrays of 
the same datatype.

Note to future: if mixed data-types should be implemented (e.g. look at isin_set_long_in_ushort)
you have to ensure that the underlying unordered list is of a data-type can represent the
full range of both used data-dtypes.

There is a generic functions (isin_set) and some type specific ones (isin_set_datatype. Both
seem to be faster then numpy.isin but the generic ones (defined with fused datatypes) seem to
introduce a bit of an overhead - even when called with data-type definition
(e.g. isin_set[cython.ushort](a,b). So the type specific ones run somewhat faster.

some timings (run on my desktop):
------------------------------------
import numpy as np
from timeit import timeit
from xicommon.cython import *
import cython

a=np.random.randint(1000,size=100,dtype=np.uint16)
b=np.random.randint(1000,size=100,dtype=np.uint16)
p=print
p("np.isin                  : " +str(timeit("np.isin(a,b)",globals=globals())))
p("np.all(np.equal):"+str(timeit("np.all(np.equal(a.reshape(-1,1),b),axis=1)",globals=globals())))
p("isin_set[cython.ushort]  : " + str(timeit("isin_set[cython.ushort](a,b)", globals=globals())))
p("isin_set                 : " + str(timeit("isin_set(a,b)", globals=globals())))
p("isin_set_ushort          : " + str(timeit("isin_set_ushort(a,b)", globals=globals())))
-----Result------------------------
np.isin                  : 37.82453854009509
np.all(np.equal(reshape)): 9.77861992828548
isin_set[cython.ushort]  : 10.08045575208962
isin_set                 : 11.699206279590726
isin_set_ushort          : 9.570891827344894
------------------------------------

The main advantage over  `np.all(np.equal(` is not so much speed as it is the amount of used
memory for largish vectors.
"""
import numpy as np
cimport numpy as np
cimport cython
from libcpp.unordered_set cimport unordered_set

"""
cython can emulate c stile generics with fused data-types. Using these here.
But while in c the actual used implementation is decided on compile time in cython there is a 
runtime decision done what method to use at every call. Meaning for each call there is a overhead for 
figuring out what version of the function to use. This can be reduced by supplying the method 
with a datatype. E.g. x=myfunc[cython.ushort](a,b).
"""
ctypedef fused set_types:
    char
    int
    short
    long
    long long
    unsigned char
    unsigned int
    unsigned short
    unsigned long
    unsigned long long
    float
    double


def isin_set_long(long[:] elements, long[:] test_elements):
    """
    Check which elements in the first vector are found in the second vector.

    Internally the second vector is converted into an unordered set to speed up the process.

    :param elements - numbers to be checked igf they are in test_elements
    :param test_elements - numbers that the numbers elements are test against
    :return: (ndarray[bool]) - boolean array indicating which numbers in the elements array are
    also in the test_elements array (True: present; False: not present)
    """
    cdef unordered_set[long] test_set
    cdef np.ndarray return_list_array = np.empty(elements.shape[0], dtype=np.bool_)
    cdef np.npy_bool[:] return_list = return_list_array
    cdef int e
    with nogil, cython.boundscheck(False), cython.wraparound(False), \
            cython.initializedcheck(False), cython.cdivision(True):
        for e in range(test_elements.shape[0]):
            test_set.insert(test_elements[e])

        for e in range(elements.shape[0]):
            return_list[e] = not (test_set.find(elements[e]) == test_set.end())

    return return_list_array

def isin_set_schar(signed char[:] elements, signed char[:] test_elements):
    """
    Check which elements in the first vector are found in the second vector.

    Internally the second vector is converted into an unordered set to speed up the process.

    :param elements - numbers to be checked igf they are in test_elements
    :param test_elements - numbers that the numbers elements are test against
    :return: (ndarray[bool]) - boolean array indicating which numbers in the elements array are
    also in the test_elements array (True: present; False: not present)
    """
    cdef unordered_set[signed char] test_set
    cdef np.ndarray return_list_array = np.empty(elements.shape[0], dtype=np.bool_)
    cdef np.npy_bool[:] return_list = return_list_array
    cdef int e
    with nogil, cython.boundscheck(False), cython.wraparound(False), \
            cython.initializedcheck(False), cython.cdivision(True):
        for e in range(test_elements.shape[0]):
            test_set.insert(test_elements[e])

        for e in range(elements.shape[0]):
            return_list[e] = not (test_set.find(elements[e]) == test_set.end())

    return return_list_array


def isin_set_ushort(unsigned short[:] elements, unsigned short[:] test_elements):
    """
    Check which elements in the first vector are found in the second vector.

    Internally the second vector is converted into an unordered set to speed up the process.

    :param elements - numbers to be checked igf they are in test_elements
    :param test_elements - numbers that the numbers elements are test against
    :return: (ndarray[bool]) - boolean array indicating which numbers in the elements array are
    also in the test_elements array (True: present; False: not present)
    """
    cdef unordered_set[unsigned short] test_set
    cdef np.ndarray return_list_array = np.empty(elements.shape[0], dtype=np.bool_)
    cdef np.npy_bool[:] return_list = return_list_array
    cdef int e
    with nogil, cython.boundscheck(False), cython.wraparound(False), \
            cython.initializedcheck(False), cython.cdivision(True):
        for e in range(test_elements.shape[0]):
            test_set.insert(test_elements[e])

        for e in range(elements.shape[0]):
            return_list[e] = not (test_set.find(elements[e]) == test_set.end())

    return return_list_array


def isin_set_double(double[:] elements, double[:] test_elements):
    """
    Check which elements in the first vector are found in the second vector.

    Internally the second vector is converted into an unordered set to speed up the process.

    :param elements - numbers to be checked igf they are in test_elements
    :param test_elements - numbers that the numbers elements are test against
    :return: (ndarray[bool]) - boolean array indicating which numbers in the elements array are
    also in the test_elements array (True: present; False: not present)
    """
    cdef unordered_set[double] test_set
    cdef np.ndarray return_list_array = np.empty(elements.shape[0], dtype=np.bool_)
    cdef np.npy_bool[:] return_list = return_list_array
    cdef int e
    with nogil, cython.boundscheck(False), cython.wraparound(False), \
            cython.initializedcheck(False), cython.cdivision(True):
        for e in range(test_elements.shape[0]):
            test_set.insert(test_elements[e])

        for e in range(elements.shape[0]):
            return_list[e] = not (test_set.find(elements[e]) == test_set.end())

    return return_list_array

def isin_set(set_types[:] elements, set_types[:] test_elements):
    """
    Check which elements in the first vector are found in the second vector.

    Internally the second vector is converted into an unordered set to speed up the process.

    NOTE: using the fused datatype seems to introduce a bit of an overhead - as using statically
    type method seem to be faster then this one.

    :param elements - numbers to be checked igf they are in test_elements
    :param test_elements - numbers that the numbers elements are test against
    :return: (ndarray[bool]) - boolean array indicating which numbers in the elements array are
    also in the test_elements array (True: present; False: not present)
    """
    cdef unordered_set[set_types] test_set
    cdef np.ndarray return_list_array = np.empty(elements.shape[0], dtype=np.bool_)
    cdef np.npy_bool[:] return_list = return_list_array
    cdef int e
    with nogil, cython.boundscheck(False), cython.wraparound(False), \
            cython.initializedcheck(False), cython.cdivision(True):
        for e in range(test_elements.shape[0]):
            test_set.insert(test_elements[e])

        for e in range(elements.shape[0]):
            return_list[e] = not (test_set.find(elements[e]) == test_set.end())

    return return_list_array

def isin_set_long_in_ushort(long[:] elements, unsigned short[:] test_elements):
    """
    Check which elements in the first vector are found in the second vector.

    Internally the second vector is converted into an unordered set to speed up the process.

    :param elements - numbers to be checked igf they are in test_elements
    :param test_elements - numbers that the numbers elements are test against
    :return: (ndarray[bool]) - boolean array indicating which numbers in the elements array are
    also in the test_elements array (True: present; False: not present)
    """
    cdef unordered_set[long] test_set
    cdef np.ndarray return_list_array = np.empty(elements.shape[0], dtype=np.bool_)
    cdef np.npy_bool[:] return_list = return_list_array
    cdef int e
    with nogil, cython.boundscheck(False), cython.wraparound(False), \
            cython.initializedcheck(False), cython.cdivision(True):
        for e in range(test_elements.shape[0]):
            test_set.insert(test_elements[e])

        for e in range(elements.shape[0]):
            return_list[e] = not (test_set.find(elements[e]) == test_set.end())

    return return_list_array

# export both functions
exports = [isin_set_long, isin_set_schar, isin_set_double,
           isin_set_long_in_ushort, isin_set_ushort, isin_set]
