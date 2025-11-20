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


class DenoiseFilter(BaseFilter):
    """
    Filter to denoise a spectrum.

    Picking the n highest intensity peaks per defined m/z bin (jumping window).
    """

    def __init__(self, context, denoise_setting):
        """
        Initialise the DenoiseFilter.

        :param context: (Searcher) search context (including the config)
        :param denoise_setting: (str) key of the denoise setting to use from the config.
            (e.g. denoise_alpha, or denoise_alpha_beta)
        """
        BaseFilter.__init__(self, context)
        self.denoise_config = getattr(self.config, denoise_setting)

    def process(self, spectrum):
        """
        Process a spectrum, returning a denoised version.

        :param spectrum: (Spectrum) Spectrum to denoise
        :return: (Spectrum) Denoised copy of the spectrum
        """

        # if the spectrum is not isotope cluster resolved - work with peaks
        if spectrum.isotope_cluster_charge_values is None:
            mz_values = spectrum.mz_values
            int_values = spectrum.int_values
        else:
            # otherwise work with the cluster values
            mz_values = spectrum.isotope_cluster_mz_values
            int_values = spectrum.isotope_cluster_intensity_values

        # array of max values per bin
        bin_size = self.denoise_config.bin_size
        bins = np.arange(bin_size, np.amax(mz_values, initial=0), bin_size)
        # get the bin index for each peak
        bin_index_of_peaks = np.digitize(mz_values, bins)
        bin_selections = []
        for bin_index in range(len(bins) + 1):
            # find which peaks are in the current bin
            peak_index_in_bin = np.nonzero(bin_index_of_peaks == bin_index)[0]
            # get their corresponding intensities
            intensities = int_values[peak_index_in_bin]
            # get the indices (relative to peak_index_in_bin) of the
            # sorted intensities and pick last n
            selected_peaks = np.argsort(intensities)[-self.denoise_config.top_n:]
            # get the relevant peaks by the selected indices
            bin_selections.append(peak_index_in_bin[selected_peaks])

        s = np.concatenate(bin_selections)
        new_spec = copy(spectrum)
        sort_mask = np.argsort(mz_values[s])

        # was it a isotope cluster resolved spectrum
        if spectrum.isotope_cluster_charge_values is None:
            # No - so just replace the peaks
            new_spec.mz_values = mz_values[s][sort_mask]
            new_spec.int_values = int_values[s][sort_mask]
        else:
            # it was isotope resolved - so we can just replace the isotope information
            new_spec.isotope_cluster_mz_values = mz_values[s][sort_mask]
            new_spec.isotope_cluster_int_values = int_values[s][sort_mask]
            new_spec.isotope_cluster_charge_values = spectrum.isotope_cluster_charge_values[s][
                sort_mask]

        return new_spec
