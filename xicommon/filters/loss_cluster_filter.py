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

import numpy as np
import copy
from xicommon.filters.base_filter import BaseFilter


class LossClusterFilter(BaseFilter):
    """Remove neutral loss matching isotope cluster peaks from isotope reduced spectrum."""

    def process(self, spectrum):
        """
        Process a spectrum, removing loss matching isotope cluster peaks.

        Annotated isotope cluster peaks that match to a neutral loss of another isotope cluster are
        removed from the spectrum.
        ToDo: Add loss cluster intensity to the matched primary cluster?
        ToDo: minimal intensity of match? or min ratio?
        :param spectrum: (Spectrum) Input spectrum
        :return: (Spectrum) Processed copy of the input spectrum.
        """
        # check if input spectrum has charge annotations
        if spectrum.isotope_cluster_charge_values is None:
            raise ValueError
        if len(self.config.fragmentation.losses) == 0:
            return copy.deepcopy(spectrum)
        output_spectrum = copy.copy(spectrum)
        mz_values = spectrum.isotope_cluster_mz_values
        charge_values = spectrum.isotope_cluster_charge_values
        int_values = spectrum.isotope_cluster_intensity_values

        defined_charge_mask = charge_values != 0
        cluster_mzs = mz_values[defined_charge_mask]
        cluster_charges = charge_values[defined_charge_mask]
        cluster_ints = int_values[defined_charge_mask]

        # hstack all possible loss mzs for all losses
        loss_mass_arr = np.hstack([cluster_mzs - loss.mass / cluster_charges
                                   for loss in self.config.fragmentation.losses])
        mz_matches = np.isclose(cluster_mzs.reshape(-1, 1), loss_mass_arr,
                                rtol=self.context.get_ms2_rtol(spectrum.file_name),
                                atol=self.context.get_ms2_atol(spectrum.file_name))

        charge_matches = np.equal(cluster_charges.reshape(-1, 1), cluster_charges)

        # for a match mz and charge of a cluster peak must match to any loss peak
        loss_match_mask = np.any(mz_matches, 1) & np.any(charge_matches, 1)

        # result peaks are the undefined charge peaks + the filtered cluster peaks
        out_mz_values = np.concatenate(
            [mz_values[~defined_charge_mask], cluster_mzs[~loss_match_mask]])
        out_int_values = np.concatenate(
            [int_values[~defined_charge_mask], cluster_ints[~loss_match_mask]])
        out_charge_values = np.concatenate(
            [charge_values[~defined_charge_mask], cluster_charges[~loss_match_mask]])

        # sort the arrays by mz
        mz_sort_indices = np.argsort(out_mz_values)
        output_spectrum.isotope_cluster_mz_values = out_mz_values[mz_sort_indices]
        output_spectrum.isotope_cluster_intensity_values = out_int_values[mz_sort_indices]
        output_spectrum.isotope_cluster_charge_values = out_charge_values[mz_sort_indices]

        return output_spectrum
