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

"""Module containing the base class for all spectrum filters."""
import abc


class BaseFilter(abc.ABC):
    """Base class for spectrum filters."""

    config_needed = True

    def __init__(self, context=None):
        """Initialise the Filter."""
        if self.config_needed and (context is None):
            raise TypeError('Config is required for this filter')
        self.config = getattr(context, 'config', None)
        self.context = context

    @abc.abstractmethod
    def process(self, spectrum):
        """
        Apply the current filter to the given spectrum.

        :param spectrum: (Spectrum) The original spectrum to be processed
        :return: (Spectrum) New processed spectrum
        """
        pass
