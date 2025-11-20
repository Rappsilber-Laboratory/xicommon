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
from copy import copy
from xicommon.filters.base_filter import BaseFilter
from xicommon import const


class ChargeReducer(BaseFilter):
    """
    Reduce peaks in a charge annotated spectrum to singly charged peaks.

    Peaks with charges >1 have their m/z value multiplied by their charge state.
    Peaks with unknown charge state (0) are unaffected.
    """

    def process(self, spectrum):
        """
        Process a spectrum, returning a charged reduced version.

        :param spectrum: (Spectrum) Spectrum to process
        :return: (Spectrum) charged reduced copy of the spectrum
        """
        if spectrum.isotope_cluster_charge_values is None:
            raise ValueError

        # calculate charge reduced mz
        mz_values = spectrum.isotope_cluster_mz_values
        charge_values = spectrum.isotope_cluster_charge_values
        charge_reduced_mz = (mz_values - const.PROTON_MASS) * charge_values + const.PROTON_MASS

        # set undefined charge values to initial m/z values
        undefined_charge_mask = charge_values == 0
        charge_reduced_mz[undefined_charge_mask] = mz_values[undefined_charge_mask]

        new_spec = copy(spectrum)

        same_original_charge_mask = charge_values == charge_values.reshape(-1, 1)
        overlaps_without_same_charge = np.isclose(
            charge_reduced_mz,
            charge_reduced_mz.reshape(-1, 1),
            atol=self.context.get_ms2_atol(spectrum.source_path),
            rtol=self.context.get_ms2_rtol(spectrum.source_path)
        )
        overlaps = overlaps_without_same_charge & ~same_original_charge_mask

        # because `overlaps` is symmetric, we take the triangular
        # part above the diagonal and check that for any matches.
        remove_mask = np.any(np.triu(overlaps, 1), axis=1)

        # sum up all int values where there is an overlap, but not
        # when they had the same original charge, as they should
        # not be collapsed.
        res_int = np.sum((overlaps | np.identity(len(overlaps), dtype=bool))
                         * spectrum.isotope_cluster_intensity_values, 1)
        filtered_int = compress(res_int, remove_mask)

        filtered_mz = compress(charge_reduced_mz, remove_mask)

        filtered_charge = compress(~undefined_charge_mask, remove_mask)

        sort_mask = np.argsort(filtered_mz)
        new_spec.mz_values = filtered_mz[sort_mask]
        new_spec.int_values = filtered_int[sort_mask]
        new_spec.isotope_cluster_mz_values = filtered_mz[sort_mask]
        new_spec.isotope_cluster_intensity_values = filtered_int[sort_mask]
        new_spec.isotope_cluster_charge_values = filtered_charge[sort_mask].astype(int)

        return new_spec


def compress(array, mask):
    return np.ma.array(array, mask=mask).compressed()
