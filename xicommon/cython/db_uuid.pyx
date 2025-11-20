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

# cython: language_level=3
import secrets
from libc.time cimport time
import uuid

cdef unsigned long long counter=0

def db_uuid():
    """
    Generates a UUID with a prefix of 32 ones and a following timestamp.
    The purpose is to have UUIDs that are always appended at the end of the DB index.
    WARNING: Does not follow the actual UUID definition (version/variant not set) but PostgreSQL does not care
    """
    global counter

    # Generate upper 64bits (prefix and timestamp)
    cdef unsigned long long prefix = 0xFFFFFFFF00000000
    cdef unsigned long long timestamp = <unsigned int>(time(NULL) & 0xFFFFFFFF)

    cdef unsigned long long uuid_upper = prefix | timestamp

    # Generate lower 64bits (counter and random)
    cdef unsigned long long random_value = secrets.randbits(56)
    cdef unsigned long long uuid_lower = (counter << 56) | random_value

    # Increment global counter
    counter = (counter+1)%64

    # Convert integer to UUID object
    return uuid.UUID(
        int=(int(uuid_upper)<<64)|int(uuid_lower)
    )
