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
Base context class providing tolerance and error management.

This module provides ContextBase, a minimal base class for contexts that need
tolerance and error management from config recalibration settings. Both SearchContext
and MockContext extend this class.
"""


class ContextBase:
    """
    Base context providing tolerance and error management.

    This class manages file-specific and default tolerances and errors for MS1, MS2,
    and isotope detection based on configuration recalibration settings.
    """

    def __init__(self, config):
        """
        Initialize ContextBase with configuration.

        :param config: (Config) Configuration object with recalibration settings
        """
        self.config = config
        self.set_tolerances_and_errors()

    def set_tolerances_and_errors(self):
        """Create dictionary with file-specific tolerances from recalibration in config."""
        self._atolerances_ms1 = {}
        self._atolerances_ms2 = {}
        self._rtolerances_ms1 = {}
        self._rtolerances_ms2 = {}
        self._aerrors_ms1 = {}
        self._aerrors_ms2 = {}
        self._rerrors_ms1 = {}
        self._rerrors_ms2 = {}
        self._isotope_rtol = {}

        for recali_config in self.config.recalibration:
            # store file specific tolerances for fast (and easy) access if useful
            if recali_config.ms1_atol > 0:
                self._atolerances_ms1[recali_config.file] = recali_config.ms1_atol
            if recali_config.ms2_atol > 0:
                self._atolerances_ms2[recali_config.file] = recali_config.ms2_atol
            if recali_config.ms1_rtol > 0:
                self._rtolerances_ms1[recali_config.file] = recali_config.ms1_rtol
            if recali_config.ms2_rtol > 0:
                self._rtolerances_ms2[recali_config.file] = recali_config.ms2_rtol

            # store file specific errors for fast (and easy) access
            self._aerrors_ms1[recali_config.file] = recali_config.ms1_aerror
            self._aerrors_ms2[recali_config.file] = recali_config.ms2_aerror
            self._rerrors_ms1[recali_config.file] = recali_config.ms1_rerror
            self._rerrors_ms2[recali_config.file] = recali_config.ms2_rerror

            # store file specific relative tolerance as isotope tolerance if smaller than default
            if recali_config.ms2_rtol < self.config.isotope_config.rtol:
                self._isotope_rtol[recali_config.file] = recali_config.ms2_rtol

        # store default tolerances and errors for fast access
        # the instance variable is about 10 % faster than the instance instance variable
        # 86.98 ms vs 76.13 ms for 1000000 iterations
        self._default_rtolerances_ms1 = self.config.ms1_rtol
        self._default_rtolerances_ms2 = self.config.ms2_rtol
        self._default_atolerances_ms1 = self.config.ms1_atol
        self._default_atolerances_ms2 = self.config.ms2_atol
        self._default_rerrors_ms1 = 0
        self._default_rerrors_ms2 = 0
        self._default_aerrors_ms1 = 0
        self._default_aerrors_ms2 = 0
        self._default_isotope_rtol = self.config.isotope_config.rtol

    def get_ms1_rtol(self, file):
        """Get the relative tolerance for MS1 for the given file."""
        return self._rtolerances_ms1.get(file, self._default_rtolerances_ms1)

    def get_ms1_atol(self, file):
        """Get the absolute tolerance for MS1 for the given file."""
        return self._atolerances_ms1.get(file, self._default_atolerances_ms1)

    def get_ms2_rtol(self, file):
        """Get the relative tolerance for MS2 for the given file."""
        return self._rtolerances_ms2.get(file, self._default_rtolerances_ms2)

    def get_ms2_atol(self, file):
        """Get the absolute tolerance for MS2 for the given file."""
        return self._atolerances_ms2.get(file, self._default_atolerances_ms2)

    def get_isotope_rtol(self, file):
        """Get the relative tolerance for isotope detection for the given file."""
        return self._isotope_rtol.get(file, self._default_isotope_rtol)

    def get_ms1_rerror(self, file):
        """Get the relative error for MS1 for the given file."""
        return self._rerrors_ms1.get(file, self._default_rerrors_ms1)

    def get_ms1_aerror(self, file):
        """Get the absolute error for MS1 for the given file."""
        return self._aerrors_ms1.get(file, self._default_aerrors_ms1)

    def get_ms2_rerror(self, file):
        """Get the relative error for MS2 for the given file."""
        return self._rerrors_ms2.get(file, self._default_rerrors_ms2)

    def get_ms2_aerror(self, file):
        """Get the absolute error for MS2 for the given file."""
        return self._aerrors_ms2.get(file, self._default_aerrors_ms2)

    def has_recalibration(self, file):
        """Return True if there is any recalibration stored for a file."""
        return (file in self._aerrors_ms2 or file in self._rerrors_ms2
                or file in self._aerrors_ms1 or file in self._rerrors_ms1)
