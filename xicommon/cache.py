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

"""This module handles caching of data."""
import os


class CacheItem(object):
    """An individual item in the cache."""

    def __init__(self, cache, name, version, hash_value, extension):
        """
        Initialise the CacheItem.

        :param cache: (Cache) cache object this item belongs to.
        :param name: (str) Name for this item.
        :param version: (int) Version number for the format of the cached data.
        :param hash_value: (str) Hash value over all parameters used to generate this item.
        :param extension: (str) Extension to use for item filename.
        """
        self.cache = cache
        self.name = name
        self.version = int(version)
        self.hash = "%08x" % (hash_value % 0xFFFFFFFF)
        self.extension = extension

    @property
    def load_filename(self):
        """Filename to use when loading this cache item."""
        return os.path.join(self.cache.directory, "%s-%s-v%d%s" % (self.name,
                                                                   self.hash,
                                                                   self.version,
                                                                   self.extension))

    @property
    def save_filename(self):
        """Filename to use when saving this cache item."""
        return os.path.join(self.cache.directory, "%s-%s-v%d.partial%s" % (self.name,
                                                                           self.hash,
                                                                           self.version,
                                                                           self.extension))

    def validate(self):
        """Mark the data for this cache item as valid"""
        os.rename(self.save_filename, self.load_filename)

    @property
    def exists(self):
        """Whether this item already exists in the cache."""
        return os.path.exists(self.load_filename)


class Cache(object):
    """A cache for pre-prepared data."""

    def __init__(self, directory=None):
        """
        Initialise the Cache.

        :param directory: (str) Path to directory to store cached data.
        """
        if directory is None:
            directory = os.curdir
        self.directory = directory

    def item(self, name, version, hash_value, extension=""):
        """
        Return a handle for an individual cache item.

        :param name: (str) Name for this item.
        :param version: (int) Version number for the format of the cached data.
        :param hash_value: (str) Hash value over all parameters used to generate this item.
        :param extension: (str) Extension to use for item filename.
        """
        return CacheItem(self, name, version, hash_value, extension)
