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
from xicommon.filters import LossClusterFilter
from xicommon.mock_context import MockContext
from xicommon.config import Config, FragmentationConfig, Loss
import numpy as np
from numpy.testing import assert_array_equal
import pytest


def test_loss_cluster_filter():
    # set up config and context
    config = Config(fragmentation=FragmentationConfig(
        losses=[
            Loss(name='l1', mass=10.0, specificity=['K']),
            Loss(name='l2', mass=18.0, specificity=['K']),
        ])
    )
    ctx = MockContext(config)

    # init filter
    loss_cluster_filter = LossClusterFilter(ctx)

    # create test spectrum
    spectrum = Spectrum({},
                        (100.0, 200.0, 205.0, 282.0, 300.0, 309.0, 318.0),
                        (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0), 'test')

    # loss filter requires isotope annotation and should fail without it
    with pytest.raises(ValueError):
        loss_cluster_filter.process(spectrum)

    spectrum.isotope_cluster_charge_values = np.array([0, 2, 2, 0, 1, 2, 1])
    spectrum.isotope_cluster_mz_values = spectrum.mz_values
    spectrum.isotope_cluster_intensity_values = spectrum.int_values

    out = loss_cluster_filter.process(spectrum)

    # id 0: 100.0 stays - undefined charge state
    # id 1: 200.0 removed - l1 loss for charge 2 of 205.0
    # id 2: 205.0 stays - no match
    # id 3: 282.0 stays - undefined charge state
    # id 4: 300.0 removed - l2 loss for charge 1 of 318.0
    # id 5: 309.0 stays - no match (318.0 is correct m/z but wrong charge state)
    # id 6: 318.0 stays - no match
    expected_mz = [100.0, 205.0, 282.0, 309.0, 318.0]
    expected_int = [100.0, 300.0, 400.0, 600.0, 700.0]
    expected_charges = [0, 2, 0, 2, 1]

    assert_array_equal(expected_mz, out.isotope_cluster_mz_values)
    assert_array_equal(expected_int, out.isotope_cluster_intensity_values)
    assert_array_equal(expected_charges, out.isotope_cluster_charge_values)

    # do the same without losses
    config = Config(fragmentation=FragmentationConfig(
        losses=[])
    )
    ctx = MockContext(config)
    loss_cluster_filter = LossClusterFilter(ctx)
    # should basically return the original spectrum
    out = loss_cluster_filter.process(spectrum)
    assert_array_equal(spectrum.isotope_cluster_mz_values,
                       out.isotope_cluster_mz_values)
    assert_array_equal(spectrum.isotope_cluster_intensity_values,
                       out.isotope_cluster_intensity_values)
    assert_array_equal(spectrum.isotope_cluster_charge_values,
                       out.isotope_cluster_charge_values)
