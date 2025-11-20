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
from xicommon.filters import IsotopeReducer
from numpy.testing import assert_array_equal
import pytest
import numpy as np


def test_raise_on_no_charge_data():
    spectrum = Spectrum({'mz': 100, 'charge': 2}, [100, 101], [1000, 20], 'test_scan')
    with pytest.raises(ValueError):
        IsotopeReducer().process(spectrum)


def test_trivial_case():
    spectrum = Spectrum({'mz': 100, 'charge': 2}, [100, 101], [1000, 20], 'test_scan')
    spectrum.isotope_cluster_ids = np.array([0, 1])
    spectrum.isotope_cluster_charge_values = np.array([0, 0])
    spectrum.isotope_cluster_mz_values = spectrum.mz_values
    spectrum.isotope_cluster_intensity_values = spectrum.int_values
    out = IsotopeReducer().process(spectrum)
    assert_array_equal(out.mz_values, [100, 101])
    assert_array_equal(out.int_values, [1000, 20])


def test_isotope_reducer():
    spectrum = Spectrum({'mz': 100, 'charge': 2},
                        [90, 91, 92, 98, 100, 101, 102, 103],
                        [2000, 200, 20, 500, 1000, 20, 10, 5], 'test_scan')
    spectrum.isotope_cluster_charge_values = np.array([1, 2, 3])
    spectrum.isotope_cluster_mz_values = np.array([90, 98, 100])
    spectrum.isotope_cluster_intensity_values = np.array([2000, 500, 1000])
    out = IsotopeReducer().process(spectrum)
    assert_array_equal(out.mz_values, [90, 98, 100])
    assert_array_equal(out.int_values, [2000, 500, 1000])
    assert_array_equal(out.charge_values, [1, 2, 3])
