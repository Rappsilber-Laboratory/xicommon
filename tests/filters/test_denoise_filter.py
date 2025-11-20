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

from xicommon.config import Config, DenoiseConfig
from xicommon.spectra_reader import MGFReader, Spectrum
from xicommon.filters import DenoiseFilter
from numpy.testing import assert_equal, assert_array_equal
import pytest
import numpy as np
import os
from xicommon.mock_context import MockContext


@pytest.fixture
def load_mgf():
    def loader(file='test.mgf'):
        config = Config()
        ctx = MockContext(config)
        reader = MGFReader(ctx)
        current_dir = os.path.dirname(__file__)
        reader.load(os.path.join(current_dir, '..', 'fixtures', 'spectra', file))
        return reader
    return loader


def test_denoise_config_selection():
    context = MockContext(Config(
        denoise_alpha=DenoiseConfig(top_n=1, bin_size=10),
        denoise_alpha_beta=DenoiseConfig(top_n=2, bin_size=20)
    ))

    alpha_denoise_filter = DenoiseFilter(context, 'denoise_alpha')
    alpha_beta_denoise_filter = DenoiseFilter(context, 'denoise_alpha_beta')

    assert alpha_denoise_filter.denoise_config.top_n == 1
    assert alpha_denoise_filter.denoise_config.bin_size == 10

    assert alpha_beta_denoise_filter.denoise_config.top_n == 2
    assert alpha_beta_denoise_filter.denoise_config.bin_size == 20


def test_denoise_simple_spectra(load_mgf):
    reader = load_mgf()
    expected_mz_arrays = [(846.60, 847.60), (345.10, 370.20, 460.20)]
    expected_int_arrays = [(73, 67), (237, 128, 108)]

    context = MockContext(Config(denoise_alpha=DenoiseConfig(top_n=2, bin_size=100)))
    denoise_filter = DenoiseFilter(context, 'denoise_alpha')
    for i, spectrum in enumerate(reader.spectra):
        out = denoise_filter.process(spectrum)
        assert_equal(out.mz_values, expected_mz_arrays[i])
        assert_equal(out.int_values, expected_int_arrays[i])


def test_denoise_bigger_spectrum(load_mgf):
    # should this just be on the spectrum instead of testing the reader?
    # write test for mz < 100
    reader = load_mgf('bigger_spectrum.mgf')
    spectrum = next(reader.spectra)
    context = MockContext(Config(denoise_alpha_beta=DenoiseConfig(top_n=4, bin_size=100)))
    denoise_filter = DenoiseFilter(context, 'denoise_alpha_beta')
    out = denoise_filter.process(spectrum)
    expected_mz = (846.6, 846.8, 847.6, 848.01, 1272.62, 1273.12, 1283.1, 1284.12)
    expected_int = (73.0, 44.0, 67.0, 18.0, 81.0, 70.0, 112.0, 181.0)
    assert_equal(out.mz_values, expected_mz)
    assert_equal(out.int_values, expected_int)


def test_denoise_preserves_additional_spectrum_attrs():
    spectrum = Spectrum({},
                        (846.6, 846.8, 847.6, 848.01, 1272.62, 1273.12, 1283.1, 1284.12),
                        (73.0, 44.0, 67.0, 18.0, 81.0, 70.0, 112.0, 181.0), 'test')
    spectrum.isotope_cluster_charge_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    spectrum.isotope_cluster_mz_values = spectrum.mz_values
    spectrum.isotope_cluster_intensity_values = spectrum.int_values
    context = MockContext(Config(denoise_alpha=DenoiseConfig(top_n=2, bin_size=100)))
    denoise_filter = DenoiseFilter(context, 'denoise_alpha')
    out = denoise_filter.process(spectrum)
    expected_charge = np.array([1, 3, 7, 8])
    assert_equal(out.isotope_cluster_charge_values, expected_charge)

    # test that the input spectrum is still unchanged
    assert_array_equal(spectrum.mz_values, (846.6, 846.8, 847.6, 848.01, 1272.62, 1273.12, 1283.1,
                                            1284.12))
    assert_array_equal(spectrum.int_values, (73.0, 44.0, 67.0, 18.0, 81.0, 70.0, 112.0, 181.0))
    assert_array_equal(spectrum.isotope_cluster_charge_values, (1, 2, 3, 4, 5, 6, 7, 8))
