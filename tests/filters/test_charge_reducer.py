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

from xicommon.spectra_reader import Spectrum
from xicommon.filters import ChargeReducer
from xicommon.config import Config
from xicommon import const
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from xicommon.mock_context import MockContext


config = Config()
ctx = MockContext(config)


def build_isotope_reduced_spectrum(*peaklist):
    spectrum = Spectrum({'mz': 100, 'charge': 2}, [], [], 'test_scan')
    spectrum.mz_values = np.array([x[0] for x in peaklist], dtype=np.float64)
    spectrum.int_values = np.array([x[1] for x in peaklist], dtype=np.float64)
    spectrum.isotope_cluster_mz_values = spectrum.mz_values
    spectrum.isotope_cluster_charge_values = np.array([x[2] for x in peaklist], dtype=np.int8)
    spectrum.isotope_cluster_intensity_values = np.array([x[1] for x in peaklist], dtype=np.float64)
    return spectrum


def test_config_needed():
    with pytest.raises(TypeError):
        ChargeReducer(None)


def test_check_for_charge_info_on_spectrum():
    spectrum = Spectrum({}, [1], [1], 'test_scan')
    with pytest.raises(ValueError):
        ChargeReducer(ctx).process(spectrum)


def test_simple_charge_reduction():
    spectrum = build_isotope_reduced_spectrum((100, 20, 1, 0), (200 + const.PROTON_MASS, 50, 2, 1))
    reduced = ChargeReducer(ctx).process(spectrum)
    assert_array_equal(reduced.isotope_cluster_mz_values, [100, 400 + const.PROTON_MASS])
    assert_array_equal(reduced.isotope_cluster_charge_values, [1, 1])

    # test that the original spectrum is still unchanged
    assert_array_equal(spectrum.mz_values, (100, 200 + const.PROTON_MASS))
    assert_array_equal(spectrum.int_values, (20, 50))
    assert_array_equal(spectrum.isotope_cluster_intensity_values, (20, 50))
    assert_array_equal(spectrum.isotope_cluster_charge_values, (1, 2))


def test_preserve_charge_0_peaks():
    spectrum = build_isotope_reduced_spectrum(
        (100, 20, 1, 0),
        (200 + const.PROTON_MASS, 50, 2, 1),
        (220 + const.PROTON_MASS, 50, 0, 2)
    )
    reduced = ChargeReducer(ctx).process(spectrum)
    assert_array_equal(reduced.isotope_cluster_mz_values, [100, 220 + const.PROTON_MASS,
                                                           400 + const.PROTON_MASS])
    assert_array_equal(reduced.isotope_cluster_charge_values, [1, 0, 1])


def test_collapse_peaks():
    spectrum = build_isotope_reduced_spectrum(
        (90 + const.PROTON_MASS, 20, 1, 0),
        (100 + const.PROTON_MASS, 70, 3, 1),
        (120 + const.PROTON_MASS, 70, 2, 2),
        (150 + const.PROTON_MASS, 80, 2, 3),
        (220 + const.PROTON_MASS, 50, 0, 4),
        (300 + const.PROTON_MASS, 50, 1, 5)
    )
    reduced = ChargeReducer(ctx).process(spectrum)
    assert_array_equal(reduced.isotope_cluster_mz_values,
                       np.array([90, 220, 240, 300]) + const.PROTON_MASS)
    assert_array_equal(reduced.isotope_cluster_intensity_values, [20, 50, 70, 200])
    assert_array_equal(reduced.isotope_cluster_charge_values, [1, 0, 1, 1])


def test_dont_collapse_peaks_of_same_original_charge():
    spectrum = build_isotope_reduced_spectrum(
        (199.99999, 50, 1, 0),
        (200, 50, 1, 1),
        (200.00001, 50, 1, 2)
    )
    config = Config(ms2_tol='1ppm')
    ctx = MockContext(config)
    reduced = ChargeReducer(ctx).process(spectrum)
    assert_array_equal(reduced.isotope_cluster_mz_values, [199.99999, 200, 200.00001])
    assert_array_equal(reduced.isotope_cluster_intensity_values, [50, 50, 50])
    assert_array_equal(reduced.isotope_cluster_charge_values, [1, 1, 1])


def test_add_charge_reduced_peaks_to_all_matching_peaks():
    spectrum = build_isotope_reduced_spectrum(
        (75 + const.PROTON_MASS, 10, 4, 0),
        (100 + const.PROTON_MASS, 15, 3, 1),
        (149.99999 + const.PROTON_MASS, 25, 2, 2),
        (299.99999 + const.PROTON_MASS, 50, 1, 3),
        (300 + const.PROTON_MASS, 50, 1, 4),
        (300.00001 + const.PROTON_MASS, 50, 1, 5),
        (300.001 + const.PROTON_MASS, 50, 1, 6)
    )
    config = Config(ms2_tol='1ppm')
    ctx = MockContext(config)
    reduced = ChargeReducer(ctx).process(spectrum)
    assert_array_equal(reduced.isotope_cluster_mz_values,
                       np.array([299.99999, 300, 300.00001, 300.001])
                       + const.PROTON_MASS)
    assert_array_equal(reduced.isotope_cluster_intensity_values, [100, 100, 100, 50])
    assert_array_equal(reduced.isotope_cluster_charge_values, [1, 1, 1, 1])


def test_configure_collapse_tolerances():
    spectrum = build_isotope_reduced_spectrum(
        (100 + const.PROTON_MASS, 20, 2, 0),
        (200.001 + const.PROTON_MASS, 50, 1, 1),
        (110 + const.PROTON_MASS, 20, 2, 2),
        (220.0002 + const.PROTON_MASS, 50, 1, 3)
    )
    config = Config(ms2_tol='1ppm')
    ctx = MockContext(config)
    reduced = ChargeReducer(ctx).process(spectrum)
    assert_array_equal(reduced.isotope_cluster_mz_values,
                       np.array([200, 200.001, 220.0002]) + const.PROTON_MASS)
    assert_array_equal(reduced.isotope_cluster_charge_values, [1, 1, 1])
