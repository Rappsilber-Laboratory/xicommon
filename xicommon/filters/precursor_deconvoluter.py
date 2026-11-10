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
                 isolation_margin=1.0, max_candidates=3, rtol=25e-6, atol=0.008):
        """
        Initialise the PrecursorDeconvoluter.

        :param min_charge: (int) Minimum precursor charge state to consider
        :param max_charge: (int) Maximum precursor charge state to consider
        :param min_correlation: (float) Minimum envelope fit score (cosine similarity)
        :param isolation_margin: (float) Margin in Th added around isolation window
        :param max_candidates: (int) Maximum number of precursor candidates to return per scan
        :param rtol: (float) Relative m/z tolerance for matching isotope peaks (default 25 ppm)
        :param atol: (float) Absolute m/z tolerance for matching isotope peaks (default 8 mDa)
        """
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_correlation = min_correlation
        self.isolation_margin = isolation_margin
        self.max_candidates = max_candidates
        self.rtol = rtol
        self.atol = atol

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
        fact = np.array([math.factorial(i) for i in range(n_peaks)], dtype=np.float64)
        p = (lam ** k) * np.exp(-lam) / fact
        max_p = np.max(p)
        return p / max_p if max_p > 0 else p

    def deconvolute(self, mzs, ints, target_mz, iso_width=1.6):
        """
        Deconvolute precursor candidates from MS1 peaks within the isolation window.

        :param mzs: (ndarray) 1D sorted array of MS1 m/z values
        :param ints: (ndarray) 1D array of MS1 intensities
        :param target_mz: (float) Center of MS2 isolation window or precursor m/z
        :param iso_width: (float) Full width of MS2 isolation window in Th
        :return: (list[PrecursorCandidate]) List of accepted precursor candidates
        """
        if len(mzs) == 0:
            return []

        half_w = iso_width / 2.0
        win_min = target_mz - half_w - self.isolation_margin
        win_max = target_mz + half_w + self.isolation_margin

        in_win_idx = np.where((mzs >= win_min) & (mzs <= win_max))[0]
        if len(in_win_idx) == 0:
            return []

        candidates = []

        for z in range(self.min_charge, self.max_charge + 1):
            delta = self.C12C13_DIFF / z
            for start_idx in in_win_idx:
                m0 = mzs[start_idx]
                mass = (m0 - self.PROTON) * z
                if mass < 400 or mass > 25000:
                    continue

                theo_full = self.theoretical_distribution(mass, n_peaks=10)
                sig_peaks = np.where(theo_full >= 0.05)[0]
                if len(sig_peaks) < 2:
                    continue
                n_peaks = max(3, sig_peaks[-1] + 1)
                theo = theo_full[:n_peaks]

                obs = np.zeros(n_peaks)
                matched_mzs = []
                matched_k = []
                matched_indices = []

                for k in range(n_peaks):
                    exp_mz = m0 + k * delta
                    tol = max(exp_mz * self.rtol, self.atol)
                    diffs = np.abs(mzs - exp_mz)
                    min_i = np.argmin(diffs)
                    if diffs[min_i] <= tol:
                        obs[k] = ints[min_i]
                        matched_mzs.append(mzs[min_i])
                        matched_k.append(k)
                        matched_indices.append(min_i)

                # Require at least 2 matched peaks, and at least one in the first two positions
                if len(matched_k) < 2 or (0 not in matched_k and 1 not in matched_k):
                    continue

                # At least one matched peak must fall inside the strict isolation window
                in_iso = any(target_mz - half_w <= mzs[mi] <= target_mz + half_w
                             for mi in matched_indices)
                if not in_iso:
                    continue

                norm_obs = np.linalg.norm(obs)
                norm_theo = np.linalg.norm(theo)
                if norm_obs == 0 or norm_theo == 0:
                    continue
                cos_sim = float(np.dot(obs, theo) / (norm_obs * norm_theo))

                if cos_sim >= self.min_correlation:
                    est_m0s = [mz_val - k_val * delta
                               for mz_val, k_val in zip(matched_mzs, matched_k)]
                    weights = [ints[mi] for mi in matched_indices]
                    m0_refined = float(np.average(est_m0s, weights=weights))

                    cand = PrecursorCandidate(
                        mz=m0_refined,
                        charge=z,
                        intensity=float(np.sum(obs)),
                        mass=(m0_refined - self.PROTON) * z,
                        score=cos_sim,
                        n_peaks=len(matched_k),
                        matched_indices=matched_indices,
                        mono_offset=0
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
    def find_master_ms1_scan(raw_reader, ms2_scan_number):
        """
        Locate the master MS1 survey scan for an MS2 scan number.

        :param raw_reader: RawFileReaderAdapter instance
        :param ms2_scan_number: (int) Scan number of the MS2 event
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

        # Backtrack to the nearest preceding MS1 scan
        for s in range(ms2_scan_number - 1, 0, -1):
            try:
                filt = raw_reader.get_filter_for_scan_number(s)
                if int(filt.ms_order) == 1:
                    return s
            except Exception:
                continue

        return None

    def deconvolute_raw_scan(self, raw_reader, ms2_scan_number, ms1_cache=None):
        """
        Deconvolute precursor ions for an MS2 scan directly from a Thermo RAW reader.

        :param raw_reader: RawFileReaderAdapter instance
        :param ms2_scan_number: (int) Scan number of the MS2 event
        :param ms1_cache: (dict, optional) Cache mapping ms1_scan_number -> (mzs, ints)
        :return: (list[PrecursorCandidate]) Accepted precursor candidates
        """
        ms1_scan = self.find_master_ms1_scan(raw_reader, ms2_scan_number)
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

        try:
            target_mz = float(raw_reader.get_precursor_mz(ms2_scan_number))
        except Exception:
            target_mz = np.nan

        try:
            trailer_data = raw_reader.get_trailer_extra_information(ms2_scan_number)
            labels = [x.rstrip(':') for x in trailer_data.labels]
            tdict = dict(zip(labels, trailer_data.values))
            iso_width = float(tdict.get('MS2 Isolation Width', 1.6))
            if np.isnan(target_mz):
                target_mz = float(tdict.get('Monoisotopic M/Z', np.nan))
        except Exception:
            iso_width = 1.6

        if np.isnan(target_mz) or target_mz <= 0:
            return []

        return self.deconvolute(mzs, ints, target_mz, iso_width)
