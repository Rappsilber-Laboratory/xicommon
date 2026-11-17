# Copyright (C) 2026  Technische Universitaet Berlin
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

from numpy.testing import assert_almost_equal

from xicommon.config import Config, IsotopeDetectorConfig, RecalibrationConfig
from xicommon.context_base import ContextBase


def test_isotope_rtol_without_recalibrated_tolerances():
    """Test that isotope tolerance is NOT overridden when use_recalibrated_tolerances is False."""
    cfg = Config(
        recalibration=[
            RecalibrationConfig(file="sample.raw", ms2_tol="5 ppm"),
            RecalibrationConfig(file="no_ms2.raw", ms1_error="2 ppm"),
        ],
        ms2_tol="20 ppm",
        isotope_config=IsotopeDetectorConfig(rtol=15e-6),
        use_recalibrated_tolerances=False,
    )
    ctx = ContextBase(cfg)

    # Even though 5 ppm < 15 ppm, use_recalibrated_tolerances is False, so default is kept
    assert_almost_equal(ctx.get_isotope_rtol("sample.raw"), 15e-6)
    # File with unset ms2_tol (0 ppm) must not result in 0 tolerance
    assert_almost_equal(ctx.get_isotope_rtol("no_ms2.raw"), 15e-6)
    # Unknown file also gets default
    assert_almost_equal(ctx.get_isotope_rtol("other.raw"), 15e-6)


def test_isotope_rtol_with_recalibrated_tolerances():
    """Test that isotope tolerance is only overridden when use_recalibrated_tolerances is True
    and narrower than the default.
    """
    cfg = Config(
        recalibration=[
            RecalibrationConfig(file="narrower.raw", ms2_tol="5 ppm"),
            RecalibrationConfig(file="wider.raw", ms2_tol="30 ppm"),
            RecalibrationConfig(file="zero_tol.raw", ms1_error="2 ppm"),  # ms2_tol default 0 ppm
        ],
        ms2_tol="20 ppm",
        isotope_config=IsotopeDetectorConfig(rtol=15e-6),
        use_recalibrated_tolerances=True,
    )
    ctx = ContextBase(cfg)

    # Narrower than default: should use recalibrated ms2_rtol
    assert_almost_equal(ctx.get_isotope_rtol("narrower.raw"), 5e-6)
    # Wider than default: should keep default isotope rtol
    assert_almost_equal(ctx.get_isotope_rtol("wider.raw"), 15e-6)
    # Zero tolerance: should keep default isotope rtol, not 0
    assert_almost_equal(ctx.get_isotope_rtol("zero_tol.raw"), 15e-6)
    # Unknown file gets default
    assert_almost_equal(ctx.get_isotope_rtol("unknown.raw"), 15e-6)
