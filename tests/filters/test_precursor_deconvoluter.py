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

import os

import numpy as np
import pytest

from xicommon import const
from xicommon.config import Config
from xicommon.filters.precursor_deconvoluter import (
    PrecursorCandidate,
    PrecursorDeconvoluter,
)
from xicommon.mock_context import MockContext
from xicommon.spectra_reader import RAWReader

fixtures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures', 'spectra')


def test_precursor_candidate_properties():
    cand = PrecursorCandidate(
        mz=605.6664,
        charge=3,
        intensity=2478843.0,
        mass=1813.9774,
        score=0.88,
        n_peaks=3,
        matched_indices={10, 11, 12},
        mono_offset=0
    )
    assert cand.mz == 605.6664
    assert cand.charge == 3
    assert cand.mass == 1813.9774
    assert cand.n_peaks == 3
    assert cand.score == 0.88
    d = cand.as_dict()
    assert d['mz'] == 605.6664
    assert d['charge'] == 3
    assert "PrecursorCandidate" in repr(cand)


def test_theoretical_distribution():
    deconv = PrecursorDeconvoluter()
    theo_1000 = deconv.theoretical_distribution(1000.0, n_peaks=6)
    assert len(theo_1000) == 6
    assert np.isclose(theo_1000[0], 1.0)
    assert theo_1000[1] < 1.0

    theo_4000 = deconv.theoretical_distribution(4000.0, n_peaks=8)
    assert len(theo_4000) == 8
    # For 4000 Da, base peak should be M+1 or M+2, not M0
    max_idx = np.argmax(theo_4000)
    assert max_idx >= 1


def test_single_envelope_deconvolution():
    deconv = PrecursorDeconvoluter(min_charge=2, max_charge=4, min_correlation=0.7)
    # Synthetic 2+ peptide at m/z 700.0, neutral mass ~ 1398 Da
    z = 2
    delta = const.C12C13_MASS_DIFF / z
    true_m0 = 700.0000
    theo = deconv.theoretical_distribution(1398.0, n_peaks=4)

    mzs = np.array([true_m0 + k * delta for k in range(4)])
    ints = theo * 1e6

    cands = deconv.deconvolute(mzs, ints, target_mz=700.5, iso_width=2.0)
    assert len(cands) >= 1
    top = cands[0]
    assert top.charge == 2
    assert np.isclose(top.mz, true_m0, atol=1e-3)
    assert top.score >= 0.95
    assert top.n_peaks >= 3


def test_chimeric_precursor_mixture():
    deconv = PrecursorDeconvoluter(min_charge=2, max_charge=5, min_correlation=0.7)

    # Co-isolated species inside isolation window [999.0, 1001.0] (center 1000.0, width 2.0)
    # Peptide 1: z=2, m0 = 999.50
    # Peptide 2: z=3, m0 = 1000.10
    d2 = const.C12C13_MASS_DIFF / 2
    d3 = const.C12C13_MASS_DIFF / 3

    mzs_p1 = [999.50 + k * d2 for k in range(3)]
    theo1 = deconv.theoretical_distribution(1997.0, n_peaks=3)
    ints_p1 = theo1 * 5e5

    mzs_p2 = [1000.10 + k * d3 for k in range(4)]
    theo2 = deconv.theoretical_distribution(2997.0, n_peaks=4)
    ints_p2 = theo2 * 1e6

    all_mzs = np.array(mzs_p1 + mzs_p2)
    all_ints = np.array(list(ints_p1) + list(ints_p2))
    sort_idx = np.argsort(all_mzs)
    all_mzs = all_mzs[sort_idx]
    all_ints = all_ints[sort_idx]

    cands = deconv.deconvolute(all_mzs, all_ints, target_mz=1000.0, iso_width=2.0)
    charges = [c.charge for c in cands]
    assert 2 in charges
    assert 3 in charges


def test_harmonic_charge_disambiguation():
    deconv = PrecursorDeconvoluter(min_charge=2, max_charge=4, min_correlation=0.65)
    # A 4+ peptide has peaks at every 0.25 Th
    z = 4
    delta = const.C12C13_MASS_DIFF / z
    true_m0 = 800.0
    theo = deconv.theoretical_distribution(3196.0, n_peaks=5)
    mzs = np.array([true_m0 + k * delta for k in range(5)])
    ints = theo * 1e6

    cands = deconv.deconvolute(mzs, ints, target_mz=800.5, iso_width=1.6)
    # Candidate should be identified as z=4, not erroneously superseded by z=2
    assert len(cands) >= 1
    top = cands[0]
    assert top.charge == 4
    assert np.isclose(top.mz, true_m0, atol=1e-3)


def test_empty_and_noise_peaks():
    deconv = PrecursorDeconvoluter()
    assert deconv.deconvolute(np.array([]), np.array([]), 500.0, 1.6) == []

    # Noise peaks with irregular spacings
    noise_mzs = np.array([500.1, 500.22, 500.67, 501.4])
    noise_ints = np.array([100.0, 150.0, 80.0, 90.0])
    cands = deconv.deconvolute(noise_mzs, noise_ints, 500.5, 1.6)
    assert len(cands) == 0


def test_deconvolute_raw_scan_orbitrap():
    raw_path = os.path.join(fixtures_dir, 'MS2_MS1_orbitrap.raw')
    if not os.path.exists(raw_path):
        pytest.skip("MS2_MS1_orbitrap.raw fixture not found")

    try:
        from native_fisher_py.raw_file_reader import RawFileReaderAdapter
    except ImportError:
        pytest.skip("native_fisher_py not available")

    raw_reader = RawFileReaderAdapter.file_factory(raw_path)
    deconv = PrecursorDeconvoluter()
    ms1_cache = {}

    # Scan 6: Master scan 1, picked precursor was 954.4949 (M+1), true mono is 954.2440 (z=4)
    cands6 = deconv.deconvolute_raw_scan(raw_reader, 6, ms1_cache=ms1_cache)
    assert len(cands6) >= 1
    top6 = cands6[0]
    assert top6.charge == 4
    assert np.isclose(top6.mz, 954.2440, atol=0.01)

    # Scan 8: Master scan 5, mono 605.6661 (z=3)
    cands8 = deconv.deconvolute_raw_scan(raw_reader, 8, ms1_cache=ms1_cache)
    assert len(cands8) >= 1
    top8 = cands8[0]
    assert top8.charge == 3
    assert np.isclose(top8.mz, 605.6661, atol=0.01)


def test_raw_reader_with_deconvolution_enabled():
    raw_path = os.path.join(fixtures_dir, 'MS2_MS1_orbitrap.raw')
    if not os.path.exists(raw_path):
        pytest.skip("MS2_MS1_orbitrap.raw fixture not found")

    cfg = Config()
    cfg.precursor_deconvolution.enabled = True
    ctx = MockContext(cfg)

    reader = RAWReader(ctx)
    reader.load(raw_path)

    # When deconvolution is enabled, _convert_spectra should return deconvoluted candidates
    specs = reader._convert_spectra(6)
    assert len(specs) >= 1
    assert specs[0].precursor['charge'] == 4
    assert np.isclose(specs[0].precursor['mz'], 954.2440, atol=0.01)
    assert specs[0].precursor.get('deconvoluted') is True

    # Test spectra generator yields deconvoluted spectra
    count = 0
    for s in reader.spectra:
        count += 1
        if count >= 5:
            break
    assert count >= 1


def test_cython_python_equivalence():
    deconv_py = PrecursorDeconvoluter(min_charge=2, max_charge=5, min_correlation=0.7,
                                      use_cython=False)
    deconv_cy = PrecursorDeconvoluter(min_charge=2, max_charge=5, min_correlation=0.7,
                                      use_cython=True)

    d2 = const.C12C13_MASS_DIFF / 2
    d3 = const.C12C13_MASS_DIFF / 3

    mzs_p1 = [999.50 + k * d2 for k in range(3)]
    theo1 = deconv_py.theoretical_distribution(1997.0, n_peaks=3)
    ints_p1 = theo1 * 5e5

    mzs_p2 = [1000.10 + k * d3 for k in range(4)]
    theo2 = deconv_py.theoretical_distribution(2997.0, n_peaks=4)
    ints_p2 = theo2 * 1e6

    all_mzs = np.array(mzs_p1 + mzs_p2)
    all_ints = np.array(list(ints_p1) + list(ints_p2))
    sort_idx = np.argsort(all_mzs)
    all_mzs = all_mzs[sort_idx]
    all_ints = all_ints[sort_idx]

    cands_py = deconv_py.deconvolute(all_mzs, all_ints, target_mz=1000.0, iso_width=2.0)
    cands_cy = deconv_cy.deconvolute(all_mzs, all_ints, target_mz=1000.0, iso_width=2.0)

    assert len(cands_py) == len(cands_cy)
    for c_py, c_cy in zip(cands_py, cands_cy):
        assert c_py.charge == c_cy.charge
        assert np.isclose(c_py.mz, c_cy.mz, atol=1e-3)
        assert np.isclose(c_py.score, c_cy.score, atol=1e-3)
