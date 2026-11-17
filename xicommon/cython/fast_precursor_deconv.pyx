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

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt, exp, fabs

cdef double C12C13_DIFF = 1.0033548378
cdef double PROTON = 1.00727646677

cdef double[16] C_FACTS

cdef void init_facts() noexcept nogil:
    cdef int i
    C_FACTS[0] = 1.0
    for i in range(1, 16):
        C_FACTS[i] = C_FACTS[i - 1] * i

init_facts()

cdef inline int binary_search_left(const double* arr, int n, double val) noexcept nogil:
    cdef int left = 0
    cdef int right = n
    cdef int mid
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < val:
            left = mid + 1
        else:
            right = mid
    return left

cdef struct CandStruct:
    double mz
    int charge
    double intensity
    double mass
    double score
    int n_peaks
    int start_local_idx
    int mono_offset
    int matched_idx[16]

def cy_deconvolute(const double[::1] mzs,
                   const double[::1] ints,
                   double target_mz,
                   double iso_width=1.6,
                   double margin=1.0,
                   int min_charge=2,
                   int max_charge=8,
                   double rtol=25e-6,
                   double atol=0.008,
                   double min_correlation=0.65,
                   int max_candidates=3,
                   candidate_cls=None):
    """
    Fast Cython precursor deconvolution inside isolation window.

    :param mzs: 1D contiguous float64 array of sorted MS1 m/z values
    :param ints: 1D contiguous float64 array of MS1 intensity values
    :param target_mz: Precursor isolation center m/z
    :param iso_width: Isolation window width in Th
    :param margin: Extra margin around isolation window
    :param min_charge: Minimum candidate charge
    :param max_charge: Maximum candidate charge
    :param rtol: Relative mass tolerance (ppm)
    :param atol: Absolute mass tolerance (Da)
    :param min_correlation: Minimum cosine similarity to theoretical distribution
    :param max_candidates: Maximum number of precursor candidates to return
    :param candidate_cls: Optional class to construct precursor candidates (e.g. PrecursorCandidate)
    :return: list of candidate objects or dictionaries
    """
    cdef int n_total = mzs.shape[0]
    if n_total < 2:
        return []

    cdef double half_w = iso_width / 2.0
    cdef double win_min = target_mz - half_w - margin
    cdef double win_max = target_mz + half_w + margin
    cdef double iso_min = target_mz - half_w
    cdef double iso_max = target_mz + half_w

    cdef const double* p_mzs = &mzs[0]
    cdef const double* p_ints = &ints[0]

    cdef int i_start = binary_search_left(p_mzs, n_total, win_min)
    cdef int i_end = binary_search_left(p_mzs, n_total, win_max)
    cdef int n_win = i_end - i_start

    if n_win < 2:
        return []

    cdef const double* w_mzs = p_mzs + i_start
    cdef const double* w_ints = p_ints + i_start

    cdef CandStruct candidates[64]
    cdef int n_cands = 0

    cdef int z, idx, k, n_peaks, p_left, p_right, best_p, n_matched
    cdef double delta, m0, mass, lam, max_theo, exp_mz, tol
    cdef double obs[12]
    cdef double theo[12]
    cdef double matched_mz_arr[12]
    cdef int matched_k_arr[12]
    cdef int matched_p_arr[12]
    cdef double dot, sum_sq_obs, sum_sq_theo, norm_obs, norm_theo, cos_sim
    cdef double sum_weights, sum_weighted_m0, m0_refined, total_obs_int
    cdef bint has_iso_peak, has_p0, has_p1
    cdef double diff_val, min_diff

    # Cython nogil loop
    with nogil:
        for z in range(min_charge, max_charge + 1):
            delta = C12C13_DIFF / z

            for idx in range(n_win):
                m0 = w_mzs[idx]
                mass = (m0 - PROTON) * z
                if mass < 400.0 or mass > 25000.0:
                    continue

                # Fast check: must have neighbor at m0 + delta
                tol = (m0 + delta) * rtol
                if tol < atol:
                    tol = atol
                p_left = binary_search_left(w_mzs, n_win, (m0 + delta) - tol)
                p_right = binary_search_left(w_mzs, n_win, (m0 + delta) + tol)
                if p_right <= p_left:
                    continue

                # Theoretical Poisson distribution
                lam = mass * 0.000595
                if lam < 0.01:
                    lam = 0.01
                n_peaks = <int>(lam + 3.0 * sqrt(lam))
                if n_peaks < 3:
                    n_peaks = 3
                if n_peaks > 10:
                    n_peaks = 10

                max_theo = 0.0
                for k in range(n_peaks):
                    theo[k] = (lam ** k) * exp(-lam) / C_FACTS[k]
                    if theo[k] > max_theo:
                        max_theo = theo[k]

                if max_theo > 0.0:
                    for k in range(n_peaks):
                        theo[k] /= max_theo

                # Match peaks
                n_matched = 0
                has_iso_peak = False
                has_p0 = False
                has_p1 = False
                total_obs_int = 0.0

                for k in range(n_peaks):
                    obs[k] = 0.0
                    exp_mz = m0 + k * delta
                    tol = exp_mz * rtol
                    if tol < atol:
                        tol = atol

                    p_left = binary_search_left(w_mzs, n_win, exp_mz - tol)
                    p_right = binary_search_left(w_mzs, n_win, exp_mz + tol)

                    if p_right > p_left:
                        if p_right - p_left == 1:
                            best_p = p_left
                        else:
                            best_p = p_left
                            min_diff = fabs(w_mzs[p_left] - exp_mz)
                            for p_i in range(p_left + 1, p_right):
                                diff_val = fabs(w_mzs[p_i] - exp_mz)
                                if diff_val < min_diff:
                                    min_diff = diff_val
                                    best_p = p_i

                        obs[k] = w_ints[best_p]
                        total_obs_int += obs[k]
                        matched_mz_arr[n_matched] = w_mzs[best_p]
                        matched_k_arr[n_matched] = k
                        matched_p_arr[n_matched] = best_p
                        n_matched += 1

                        if k == 0:
                            has_p0 = True
                        if k == 1:
                            has_p1 = True
                        if w_mzs[best_p] >= iso_min and w_mzs[best_p] <= iso_max:
                            has_iso_peak = True

                if n_matched < 2 or not (has_p0 or has_p1) or not has_iso_peak:
                    continue

                # Cosine similarity
                dot = 0.0
                sum_sq_obs = 0.0
                sum_sq_theo = 0.0
                for k in range(n_peaks):
                    dot += obs[k] * theo[k]
                    sum_sq_obs += obs[k] * obs[k]
                    sum_sq_theo += theo[k] * theo[k]

                if sum_sq_obs == 0.0 or sum_sq_theo == 0.0:
                    continue

                norm_obs = sqrt(sum_sq_obs)
                norm_theo = sqrt(sum_sq_theo)
                cos_sim = dot / (norm_obs * norm_theo)

                if cos_sim >= min_correlation:
                    # Centroid recalibration
                    sum_weights = 0.0
                    sum_weighted_m0 = 0.0
                    for k in range(n_matched):
                        best_p = matched_p_arr[k]
                        sum_weights += w_ints[best_p]
                        sum_weighted_m0 += w_ints[best_p] * (matched_mz_arr[k] - matched_k_arr[k] * delta)

                    if sum_weights > 0.0:
                        m0_refined = sum_weighted_m0 / sum_weights
                    else:
                        m0_refined = m0

                    if n_cands < 64:
                        candidates[n_cands].mz = m0_refined
                        candidates[n_cands].charge = z
                        candidates[n_cands].intensity = total_obs_int
                        candidates[n_cands].mass = (m0_refined - PROTON) * z
                        candidates[n_cands].score = cos_sim
                        candidates[n_cands].n_peaks = n_matched
                        candidates[n_cands].start_local_idx = idx
                        candidates[n_cands].mono_offset = 0
                        for k in range(n_matched):
                            candidates[n_cands].matched_idx[k] = i_start + matched_p_arr[k]
                        n_cands += 1

    # Python post-processing (disambiguation and sorting)
    py_cands = []
    cdef int c_i
    for c_i in range(n_cands):
        m_indices = {candidates[c_i].matched_idx[k] for k in range(candidates[c_i].n_peaks)}
        if candidate_cls is not None:
            cand_obj = candidate_cls(
                mz=candidates[c_i].mz,
                charge=candidates[c_i].charge,
                intensity=candidates[c_i].intensity,
                mass=candidates[c_i].mass,
                score=candidates[c_i].score,
                n_peaks=candidates[c_i].n_peaks,
                matched_indices=m_indices,
                mono_offset=candidates[c_i].mono_offset
            )
        else:
            cand_obj = {
                'mz': candidates[c_i].mz,
                'charge': candidates[c_i].charge,
                'intensity': candidates[c_i].intensity,
                'mass': candidates[c_i].mass,
                'score': candidates[c_i].score,
                'n_peaks': candidates[c_i].n_peaks,
                'matched_indices': m_indices,
                'mono_offset': candidates[c_i].mono_offset
            }
        py_cands.append(cand_obj)

    if candidate_cls is not None:
        py_cands.sort(key=lambda c: (-c.n_peaks, -c.score, -c.intensity))
        accepted = []
        for cand in py_cands:
            is_dup = False
            for acc in accepted:
                if cand.charge == acc.charge:
                    if cand.matched_indices.issubset(acc.matched_indices):
                        is_dup = True
                        break
                    if abs(cand.mz - acc.mz) < 0.05:
                        is_dup = True
                        break
                if (acc.charge == 2 * cand.charge
                        and cand.matched_indices.issubset(acc.matched_indices)):
                    is_dup = True
                    break
            if not is_dup:
                accepted.append(cand)
                if len(accepted) >= max_candidates:
                    break
        return accepted
    else:
        py_cands.sort(key=lambda c: (-c['n_peaks'], -c['score'], -c['intensity']))
        accepted = []
        for cand in py_cands:
            is_dup = False
            for acc in accepted:
                if cand['charge'] == acc['charge']:
                    if cand['matched_indices'].issubset(acc['matched_indices']):
                        is_dup = True
                        break
                    if abs(cand['mz'] - acc['mz']) < 0.05:
                        is_dup = True
                        break
                if (acc['charge'] == 2 * cand['charge']
                        and cand['matched_indices'].issubset(acc['matched_indices'])):
                    is_dup = True
                    break
            if not is_dup:
                accepted.append(cand)
                if len(accepted) >= max_candidates:
                    break
        return accepted

exports = [cy_deconvolute]
