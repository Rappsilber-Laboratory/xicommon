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

from copy import copy
from xicommon.filters.base_filter import BaseFilter


class IsotopeReducer(BaseFilter):
    """Filter that reduces all detected isotope clusters to their monoisotopic peaks."""

    config_needed = False

    def process(self, spectrum):
        """
        Process a spectrum, removing all but the monoisotopic peak for detected isotope clusters.

        Input spectrum needs to have annotated isotope clusters (`isotope_cluster_ids`).
        Intensities of all other cluster peaks get added to the monoisotopic peak.
        :param spectrum: (Spectrum) Isotope detected spectrum to process
        :return: (Spectrum) Isotope reduced copy of the spectrum
        """
        # check if input spectrum is isotope detected
        if spectrum.isotope_cluster_charge_values is None:
            raise ValueError
        new_spec = copy(spectrum)

        new_spec.int_values = spectrum.isotope_cluster_intensity_values
        new_spec.mz_values = spectrum.isotope_cluster_mz_values
        new_spec.charge_values = spectrum.isotope_cluster_charge_values

        return new_spec
