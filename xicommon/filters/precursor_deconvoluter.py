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

"""Precursor deconvolution module for MS1 isolation window analysis."""

import math

import numpy as np

from xicommon import const

try:
    from xicommon.cython import cy_deconvolute
    HAVE_CYTHON = True
except (ImportError, AttributeError):
    try:
        from xicommon.cython.fast_precursor_deconv import cy_deconvolute
        HAVE_CYTHON = True
    except (ImportError, AttributeError):
        HAVE_CYTHON = False

FACTS = np.array([math.factorial(i) for i in range(15)], dtype=np.float64)


class PrecursorCandidate:
    """Represents a deconvoluted precursor ion candidate."""

    def __init__(self, mz, charge, intensity, mass, score, n_peaks,
                 matched_indices=None, mono_offset=0):
        """
        Initialise a PrecursorCandidate.

        :param mz: (float) Monoisotopic m/z (recalibrated)
        :param charge: (int) Precursor charge state
        :param intensity: (float) Total envelope or precursor intensity
        :param mass: (float) Neutral monoisotopic mass
        :param score: (float) Fit score (cosine similarity to Averagine distribution)
        :param n_peaks: (int) Number of matched isotope peaks
        :param matched_indices: (set) Indices of matched peaks in MS1 array
        :param mono_offset: (int) Isotope offset (0=true M0, 1=corrected M+1, etc.)
        """
        self.mz = float(mz)
        self.charge = int(charge)
        self.intensity = float(intensity)
        self.mass = float(mass)
        self.score = float(score)
        self.n_peaks = int(n_peaks)
        self.matched_indices = set(matched_indices) if matched_indices is not None else set()
        self.mono_offset = int(mono_offset)

    def as_dict(self):
        """Return candidate info as dictionary."""
        return {
            'mz': self.mz,
            'charge': self.charge,
            'intensity': self.intensity,
            'mass': self.mass,
            'score': self.score,
            'n_peaks': self.n_peaks,
            'mono_offset': self.mono_offset
        }

    def __repr__(self):
        return (f"PrecursorCandidate(mz={self.mz:.4f}, z={self.charge}, "
                f"mass={self.mass:.2f}, score={self.score:.2f}, n_peaks={self.n_peaks})")


class PrecursorDeconvoluter:
    """
    Deconvolutes precursor ions from MS1 survey scans within MS2 isolation windows.

    Identifies overlapping candidate isotopic envelopes, determines charge states,
    recalibrates monoisotopic m/z, and detects co-isolated chimeric precursors.
    """

    C12C13_DIFF = const.C12C13_MASS_DIFF
    PROTON = const.PROTON_MASS

    def __init__(self, min_charge=2, max_charge=8, min_correlation=0.65,
                 isolation_margin=1.0, max_candidates=3, rtol=25e-6, atol=0.008,
                 use_cython=True):
        """
        Initialise the PrecursorDeconvoluter.

        :param min_charge: (int) Minimum precursor charge state to consider
        :param max_charge: (int) Maximum precursor charge state to consider
        :param min_correlation: (float) Minimum envelope fit score (cosine similarity)
        :param isolation_margin: (float) Margin in Th added around isolation window
        :param max_candidates: (int) Maximum number of precursor candidates to return per scan
        :param rtol: (float) Relative m/z tolerance for matching isotope peaks (default 25 ppm)
        :param atol: (float) Absolute m/z tolerance for matching isotope peaks (default 8 mDa)
        :param use_cython: (bool) Whether to use fast Cython acceleration when available
        """
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_correlation = min_correlation
        self.isolation_margin = isolation_margin
        self.max_candidates = max_candidates
        self.rtol = rtol
        self.atol = atol
        self.use_cython = use_cython

    def theoretical_distribution(self, mass, n_peaks=10):
        """
        Compute theoretical Averagine isotopic distribution for a given neutral mass.

        Uses Poisson approximation of the Senko Averagine model.

        :param mass: (float) Neutral mass of the peptide/cross-link
        :param n_peaks: (int) Number of isotopic peaks to generate
        :return: (ndarray) Normalized relative peak intensities (max = 1.0)
        """
        lam = max(mass * 0.000595, 0.01)
        k = np.arange(n_peaks)
        fact = FACTS[:n_peaks]
        p = (lam ** k) * np.exp(-lam) / fact
        max_p = np.max(p)
        return p / max_p if max_p > 0 else p

    def deconvolute(self, mzs, ints, target_mz, iso_width=1.6):
        """
        Deconvolute precursor candidates from MS1 peaks within the isolation window.

        Optimized using searchsorted slicing, pairwise diff candidate pruning,
        and local binary search matching.

        :param mzs: (ndarray) 1D sorted array of MS1 m/z values
        :param ints: (ndarray) 1D array of MS1 intensities
        :param target_mz: (float) Center of MS2 isolation window or precursor m/z
        :param iso_width: (float) Full width of MS2 isolation window in Th
        :return: (list[PrecursorCandidate]) List of accepted precursor candidates
        """
        if len(mzs) < 2:
            return []

        if self.use_cython and HAVE_CYTHON:
            return cy_deconvolute(
                np.ascontiguousarray(mzs, dtype=np.float64),
                np.ascontiguousarray(ints, dtype=np.float64),
                target_mz=float(target_mz),
                iso_width=float(iso_width),
                margin=float(self.isolation_margin),
                min_charge=int(self.min_charge),
                max_charge=int(self.max_charge),
                rtol=float(self.rtol),
                atol=float(self.atol),
                min_correlation=float(self.min_correlation),
                max_candidates=int(self.max_candidates),
                candidate_cls=PrecursorCandidate,
            )

        half_w = iso_width / 2.0
        win_min = target_mz - half_w - self.isolation_margin
        win_max = target_mz + half_w + self.isolation_margin

        # Fast slice using searchsorted
        i_start = np.searchsorted(mzs, win_min)
        i_end = np.searchsorted(mzs, win_max)
        if i_end - i_start < 2:
            return []

        win_mzs = mzs[i_start:i_end]
        win_ints = ints[i_start:i_end]

        # Pairwise differences within the narrow isolation window
        diffs = win_mzs[:, None] - win_mzs[None, :]
        iso_min = target_mz - half_w
        iso_max = target_mz + half_w

        candidates = []

        for z in range(self.min_charge, self.max_charge + 1):
            delta = self.C12C13_DIFF / z
            # Candidate pruning: only test peaks that have a neighbor spaced by delta
            has_neighbor = np.any(np.abs(diffs - delta) <= max(delta * 0.05, self.atol), axis=0)
            valid_indices = np.where(has_neighbor)[0]
            if len(valid_indices) == 0:
                continue

            for idx in valid_indices:
                m0 = win_mzs[idx]
                mass = (m0 - self.PROTON) * z
                if mass < 400 or mass > 25000:
                    continue

                lam = max(mass * 0.000595, 0.01)
                n_peaks = min(8, max(3, int(lam + 3 * np.sqrt(max(lam, 1)))))
                k_arr = np.arange(n_peaks)
                theo = (lam ** k_arr) * np.exp(-lam) / FACTS[:n_peaks]
                max_theo = np.max(theo)
                if max_theo > 0:
                    theo /= max_theo

                obs = np.zeros(n_peaks)
                matched_mzs = []
                matched_k = []
                matched_local_idx = []

                for k in range(n_peaks):
                    exp_mz = m0 + k * delta
                    tol = max(exp_mz * self.rtol, self.atol)
                    p_left = np.searchsorted(win_mzs, exp_mz - tol)
                    p_right = np.searchsorted(win_mzs, exp_mz + tol)
                    if p_right > p_left:
                        if p_right - p_left == 1:
                            best_p = p_left
                        else:
                            diff_slice = np.abs(win_mzs[p_left:p_right] - exp_mz)
                            best_p = p_left + int(np.argmin(diff_slice))
                        obs[k] = win_ints[best_p]
                        matched_mzs.append(win_mzs[best_p])
                        matched_k.append(k)
                        matched_local_idx.append(best_p)

                # Require at least 2 matched peaks, and at least one in the first two positions
                if len(matched_k) < 2 or (0 not in matched_k and 1 not in matched_k):
                    continue

                # At least one matched peak must fall inside the strict isolation window
                if not any(iso_min <= win_mzs[p] <= iso_max for p in matched_local_idx):
                    continue

                norm_obs = np.linalg.norm(obs)
                norm_theo = np.linalg.norm(theo)
                if norm_obs == 0 or norm_theo == 0:
                    continue
                cos_sim = float(np.dot(obs, theo) / (norm_obs * norm_theo))

                if cos_sim >= self.min_correlation:
                    est_m0s = [mz_val - k_val * delta
                               for mz_val, k_val in zip(matched_mzs, matched_k)]
                    weights = [win_ints[p] for p in matched_local_idx]
                    m0_refined = float(np.average(est_m0s, weights=weights))
                    global_indices = {i_start + p for p in matched_local_idx}

                    mono_offset = 0
                    # Check for missing M0: for large peptides (>= 2500 Da) where M0 was below noise
                    if mass >= 2500 and len(matched_k) >= 3:
                        prev_mz = m0 - delta
                        prev_tol = max(prev_mz * self.rtol, self.atol)
                        has_prev = (np.searchsorted(win_mzs, prev_mz + prev_tol) >
                                    np.searchsorted(win_mzs, prev_mz - prev_tol))
                        if not has_prev:
                            mass_s = (m0 - delta - self.PROTON) * z
                            lam_s = max(mass_s * 0.000595, 0.01)
                            k_s = np.arange(n_peaks + 1)
                            theo_s = (lam_s ** k_s) * np.exp(-lam_s) / FACTS[:n_peaks + 1]
                            max_ts = np.max(theo_s)
                            if max_ts > 0:
                                theo_s /= max_ts
                            if theo_s[0] <= 0.45:
                                norm_ts = np.linalg.norm(theo_s[1:n_peaks + 1])
                                if norm_ts > 0 and norm_obs > 0:
                                    cos_s = float(np.dot(obs, theo_s[1:n_peaks + 1]) /
                                                  (norm_obs * norm_ts))
                                    if cos_s > cos_sim:
                                        m0_refined -= delta
                                        cos_sim = cos_s
                                        mono_offset = 1

                    cand = PrecursorCandidate(
                        mz=m0_refined,
                        charge=z,
                        intensity=float(np.sum(obs)),
                        mass=(m0_refined - self.PROTON) * z,
                        score=cos_sim,
                        n_peaks=len(matched_k),
                        matched_indices=global_indices,
                        mono_offset=mono_offset
                    )
                    candidates.append(cand)

        # Disambiguate and rank candidates
        candidates.sort(key=lambda c: (-c.n_peaks, -c.score, -c.intensity))
        accepted = []

        for cand in candidates:
            is_dup = False
            for acc in accepted:
                # Same charge: filter out sub-envelopes starting at M+1/M+2 or duplicate m/z
                if cand.charge == acc.charge:
                    if cand.matched_indices.issubset(acc.matched_indices):
                        is_dup = True
                        break
                    if abs(cand.mz - acc.mz) < 0.05:
                        is_dup = True
                        break
                # Harmonic charge check: if acc has twice the charge of cand and cand is a subset
                if (acc.charge == 2 * cand.charge
                        and cand.matched_indices.issubset(acc.matched_indices)):
                    is_dup = True
                    break
            if not is_dup:
                accepted.append(cand)
                if len(accepted) >= self.max_candidates:
                    break

        return accepted

    @staticmethod
    def find_master_ms1_scan(raw_reader, ms2_scan_number, fallback_ms1=None):
        """
        Locate the master MS1 survey scan for an MS2 scan number.

        :param raw_reader: RawFileReaderAdapter instance
        :param ms2_scan_number: (int) Scan number of the MS2 event
        :param fallback_ms1: (int, optional) Known preceding MS1 scan number
        :return: (int | None) Master MS1 scan number or None if not found
        """
        try:
            trailer_data = raw_reader.get_trailer_extra_information(ms2_scan_number)
            labels = [x.rstrip(':') for x in trailer_data.labels]
            tdict = dict(zip(labels, trailer_data.values))
            master = int(tdict.get('Master Scan Number', -1))
            total_scans = raw_reader.run_header_ex.spectra_count
            if 1 <= master <= total_scans:
                filt = raw_reader.get_filter_for_scan_number(master)
                if int(filt.ms_order) == 1:
                    return master
        except Exception:
            pass

        if fallback_ms1 is not None and fallback_ms1 > 0:
            return fallback_ms1

        # Backtrack at most 50 scans to find the preceding MS1
        min_scan = max(1, ms2_scan_number - 50)
        for s in range(ms2_scan_number - 1, min_scan - 1, -1):
            try:
                filt = raw_reader.get_filter_for_scan_number(s)
                if int(filt.ms_order) == 1:
                    return s
            except Exception:
                continue

        return None

    def deconvolute_raw_scan(self, raw_reader, ms2_scan_number, ms1_cache=None,
                             fallback_ms1=None, trailer_dict=None):
        """
        Deconvolute precursor ions for an MS2 scan directly from a Thermo RAW reader.

        :param raw_reader: RawFileReaderAdapter instance
        :param ms2_scan_number: (int) Scan number of the MS2 event
        :param ms1_cache: (dict, optional) Cache mapping ms1_scan_number -> (mzs, ints)
        :param fallback_ms1: (int, optional) Preceding MS1 scan number
        :param trailer_dict: (dict, optional) Pre-extracted trailer dictionary
        :return: (list[PrecursorCandidate]) Accepted precursor candidates
        """
        ms1_scan = self.find_master_ms1_scan(raw_reader, ms2_scan_number, fallback_ms1)
        if ms1_scan is None:
            return []

        if ms1_cache is not None and ms1_scan in ms1_cache:
            mzs, ints = ms1_cache[ms1_scan]
        else:
            try:
                stream = raw_reader.get_centroid_stream(ms1_scan, False)
                mzs = np.array(getattr(stream, "masses", []), dtype=np.float64)
                ints = np.array(getattr(stream, "intensities", []), dtype=np.float64)
            except Exception:
                return []
            if ms1_cache is not None:
                # Keep cache bounded
                if len(ms1_cache) > 10:
                    ms1_cache.pop(next(iter(ms1_cache)))
                ms1_cache[ms1_scan] = (mzs, ints)

        if len(mzs) == 0:
            return []

        if trailer_dict is None:
            try:
                trailer_data = raw_reader.get_trailer_extra_information(ms2_scan_number)
                labels = [x.rstrip(':') for x in trailer_data.labels]
                trailer_dict = dict(zip(labels, trailer_data.values))
            except Exception:
                trailer_dict = {}

        try:
            iso_width = float(trailer_dict.get('MS2 Isolation Width', 1.6))
        except (ValueError, TypeError):
            iso_width = 1.6

        target_mz = np.nan
        try:
            target_mz = float(trailer_dict.get('Monoisotopic M/Z', np.nan))
        except (ValueError, TypeError):
            pass

        if np.isnan(target_mz) or target_mz <= 0:
            try:
                target_mz = float(raw_reader.get_precursor_mz(ms2_scan_number))
            except Exception:
                target_mz = np.nan

        if np.isnan(target_mz) or target_mz <= 0:
            return []

        return self.deconvolute(mzs, ints, target_mz, iso_width)
