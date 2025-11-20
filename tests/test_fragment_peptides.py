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

from xicommon.fragment_peptides import *
from xicommon.fragmentation import spread_charges, include_losses
from xicommon.config import Config, ModificationConfig, Modification, FragmentationConfig, \
    Crosslinker, Loss
from xicommon import dtypes, const
from xicommon.cython import Fragmentation
import numpy as np
from xicommon.mock_context import MockContext
from xicommon.helper import compare_numpy


mock_frag_dtype = np.dtype([
    ('mz', np.float64), ('ion_type', '<S1'), ('idx', np.uint8), ('charge', np.uint8)
])


def test_fragment_crosslinked_peptide_pair():
    ctx = MockContext(Config(crosslinker=[Crosslinker.BS3], ms2_tol='1ppm'))
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))
    linkpos1 = 2
    linkpos2 = 0
    charge = 2

    # expected fragments
    expected_pep1_frags = fragment_crosslinked_peptide(frag_pep_index=0, cl_pep_index=1,
                                                       link_pos=linkpos1,
                                                       crosslinker=ctx.config.crosslinker[0],
                                                       context=ctx)

    expected_pep2_frags = fragment_crosslinked_peptide(frag_pep_index=1, cl_pep_index=0,
                                                       link_pos=linkpos2,
                                                       crosslinker=ctx.config.crosslinker[0],
                                                       context=ctx)
    # fragment_crosslinked peptide sets pep_id to 1 so we need to adjust this here
    expected_pep2_frags['pep_id'] = 2

    all_frags = fragment_crosslinked_peptide_pair(pep1_index=0, pep2_index=1,
                                                  link_pos1=linkpos1, link_pos2=linkpos2,
                                                  crosslinker=ctx.config.crosslinker[0],
                                                  context=ctx)
    expected_pep1_frags = spread_charges(expected_pep1_frags, ctx, charge)
    expected_pep2_frags = spread_charges(expected_pep2_frags, ctx, charge)
    expected_pep1_frags.sort(order='mz')
    expected_pep2_frags.sort(order='mz')

    all_frags = spread_charges(all_frags, ctx, charge)
    all_frags.sort(order='mz')

    pep1_frags = all_frags[all_frags['pep_id'] == 1]
    pep2_frags = all_frags[all_frags['pep_id'] == 2]

    # swap the order of ranges for pep2
    pep2_range0 = pep2_frags['ranges'][:, 0].copy()
    pep2_range1 = pep2_frags['ranges'][:, 1].copy()
    pep2_frags['ranges'][:, 0] = pep2_range1
    pep2_frags['ranges'][:, 1] = pep2_range0

    compare_numpy(expected_pep1_frags, pep1_frags)
    compare_numpy(expected_pep2_frags, pep2_frags)
    assert expected_pep1_frags.size + expected_pep2_frags.size == all_frags.size


def test_fragment_crosslinked_peptide_pair_loss():
    water_loss = Loss(name='H2O', composition="H2O1", specificity=['L'])

    ctx = MockContext(
        Config(crosslinker=Crosslinker.BS3, ms2_tol='1ppm',
               fragmentation=FragmentationConfig(losses=[water_loss], max_nloss=2))
    )
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))
    linkpos1 = 2
    linkpos2 = 0
    charge = 2

    expected_fragments = np.array([
        (65.05475797, b'b', 1, 2),  # K
        (91.54422308, b'y', 1, 2),  # Y
        (121.59678996, b'b', 2, 2),  # KL
        (129.10223948, b'b', 1, 1),  # K
        (155.59170458, b'y', 2, 2),  # KY
        (182.08116968, b'y', 1, 1),  # Y
        (191.11026147, b'y', 3, 2),  # AKY
        (242.18630346, b'b', 2, 1),  # KL
        (310.1761327, b'y', 2, 1),  # KY
        (321.69838655, b'b', 1, 2),  # L - KLM
        (357.21694345, b'b', 2, 2),  # LA - KLM
        (381.21324648, b'y', 3, 1),  # AKY
        (391.21185807, b'y', 1, 2),  # M - LAKY
        (421.26442495, b'b', 3, 2),  # LAK - KLM
        (447.75389005, b'y', 2, 2),  # LM - LAKY
        (642.38949664, b'b', 1, 1),  # L - KLM
        (713.42661043, b'b', 2, 1),  # LA - KLM
        (781.41643967, b'y', 1, 1),  # M - LAKY
        (841.52157344, b'b', 3, 1),  # LAK - KLM
        (894.50050364, b'y', 2, 1),  # LM - LAKY

        (121.59678996 - 9.005280135, b'b', 2, 2),  # KL -H2O
        (242.18630346 - 18.01056027, b'b', 2, 1),  # KL -H2O
        (321.69838655 - 9.005280135, b'b', 1, 2),  # L - KLM -H2O
        (357.21694345 - 9.005280135, b'b', 2, 2),  # LA - KLM -H2O
        (391.21185807 - 9.005280135, b'y', 1, 2),  # M - LAKY -H2O
        (421.26442495 - 9.005280135, b'b', 3, 2),  # LAK - KLM-H2O
        (447.75389005 - 9.005280135, b'y', 2, 2),  # LM - LAKY-H2O
        (642.38949664 - 18.01056027, b'b', 1, 1),  # L - KLM-H2O
        (713.42661043 - 18.01056027, b'b', 2, 1),  # LA - KLM-H2O
        (781.41643967 - 18.01056027, b'y', 1, 1),  # M - LAKY-H2O
        (841.52157344 - 18.01056027, b'b', 3, 1),  # LAK - KLM-H2O
        (894.50050364 - 18.01056027, b'y', 2, 1),  # LM - LAKY-H2O

        (321.69838655 - 18.01056027, b'b', 1, 2),  # L - KLM -H2Ox2
        (357.21694345 - 18.01056027, b'b', 2, 2),  # LA - KLM -H2Ox2
        (421.26442495 - 18.01056027, b'b', 3, 2),  # LAK - KLM-H2Ox2
        (447.75389005 - 18.01056027, b'y', 2, 2),  # LM - LAKY-H2Ox2

        (642.38949664 - 36.02112054, b'b', 1, 1),  # L - KLM-H2Ox2
        (713.42661043 - 36.02112054, b'b', 2, 1),  # LA - KLM-H2Ox2
        (841.52157344 - 36.02112054, b'b', 3, 1),  # LAK - KLM-H2Ox2
        (894.50050364 - 36.02112054, b'y', 2, 1),  # LM - LAKY-H2Ox2

    ], dtype=mock_frag_dtype)

    expected_fragments.sort(order=['mz'])

    fragments = fragment_crosslinked_peptide_pair(pep1_index=0, pep2_index=1,
                                                  link_pos1=linkpos1, link_pos2=linkpos2,
                                                  context=ctx,
                                                  crosslinker=ctx.config.crosslinker[0])
    fragments = include_losses(fragments, [0, 1], ctx)

    fragments = spread_charges(fragments, ctx, charge)

    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])


def test_fragment_crosslinked_peptide_pair_minimal():
    water_loss = Loss(name='H2O', composition="H2O1", specificity=['L'])

    ctx = MockContext(
        Config(crosslinker=Crosslinker.BS3, ms2_tol='1ppm',
               fragmentation=FragmentationConfig(losses=[water_loss], max_nloss=2))
    )
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))
    linkpos1 = [1, 2]
    linkpos2 = 0

    expected_mz = np.array([
        129.10223947,  # K
        150.05832174,  # M
        182.08116574,  # Y
        242.18630347,  # KL
        310.17612874,  # KY
        381.21324274,  # AKY
        642.38948274,  # L + KLM
        713.42659674,  # LA + KLM
        781.41643968,  # M + LAKY
        841.52155973,  # LAK + KLM
        873.54440373,  # KL + LAKY
        894.50050366,  # LM + LAKY
    ])

    expected_linearity = np.array([
        True,  # K
        True,  # M
        True,  # Y
        True,  # KL
        True,  # KY
        True,  # AKY
        False,  # L + KLM
        False,  # LA + KLM
        False,  # M + LAKY
        False,  # LAK + KLM
        False,  # KL + LAKY
        False,  # LM + LAKY
    ])

    expected_smass = np.array([0] * len(expected_mz))

    mz, linear, smass = fragment_crosslinked_peptide_pair_minimal(
        pep1_index=0, pep2_index=1, link_pos1=linkpos1, link_pos2=linkpos2,
        context=ctx, crosslinker=ctx.config.crosslinker[0])

    mzorder = np.argsort(mz)
    mz = mz[mzorder]
    linear = linear[mzorder]
    smass = smass[mzorder]

    np.testing.assert_almost_equal(expected_mz, mz, decimal=4)
    np.testing.assert_equal(expected_linearity, linear)
    np.testing.assert_equal(expected_smass, smass)


def test_fragment_noncov_peptide_pair_minimal():
    water_loss = Loss(name='H2O', composition="H2O1", specificity=['L'])

    ctx = MockContext(
        Config(crosslinker=Crosslinker.BS3, ms2_tol='1ppm',
               fragmentation=FragmentationConfig(losses=[water_loss], max_nloss=2))
    )
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))

    expected_mz = np.array([
        129.10223948,  # K
        182.08116968,  # Y
        242.18630346,  # KL
        310.1761327,  # KY
        381.21324648,  # AKY

        114.091340481879,  # L
        185.12845428687902,  # LA
        150.05832138187898,  # M
        313.223417336879,  # LAK
        263.142385396879,  # LM
    ])

    expected_mz.sort()

    expected_linearity = np.array([True] * len(expected_mz))

    expected_smass = np.array([0] * len(expected_mz))

    mz, linear, smass = fragment_noncovalent_peptide_pair_minimal(
        pep1_index=0, pep2_index=1, context=ctx)

    mzorder = np.argsort(mz)
    mz = mz[mzorder]
    linear = linear[mzorder]
    smass = smass[mzorder]

    np.testing.assert_almost_equal(expected_mz, mz, decimal=4)
    np.testing.assert_equal(expected_linearity, linear)
    np.testing.assert_equal(expected_smass, smass)


def test_fragment_crosslinked_peptide_pair_loss_precursor():
    water_loss = Loss(name='H2O', composition="H2O1", specificity=['L', 'cterm'])

    ctx = MockContext(
        Config(crosslinker=Crosslinker.BS3, ms2_tol='1ppm',
               fragmentation=FragmentationConfig(losses=[water_loss],
                                                 max_nloss=2,
                                                 add_precursor=True))
    )
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))
    linkpos1 = 2
    linkpos2 = 0
    charge = 2

    expected_fragments = np.array([
        (65.05475797, b'b', 1, 2),  # K
        (91.54422308, b'y', 1, 2),  # Y
        (121.59678996, b'b', 2, 2),  # KL
        (129.10223948, b'b', 1, 1),  # K
        (155.59170458, b'y', 2, 2),  # KY
        (182.08116968, b'y', 1, 1),  # Y
        (191.11026147, b'y', 3, 2),  # AKY
        (242.18630346, b'b', 2, 1),  # KL
        (310.1761327, b'y', 2, 1),  # KY
        (321.69838655, b'b', 1, 2),  # L - KLM
        (357.21694345, b'b', 2, 2),  # LA - KLM
        (381.21324648, b'y', 3, 1),  # AKY
        (391.21185807, b'y', 1, 2),  # M - LAKY
        (421.26442495, b'b', 3, 2),  # LAK - KLM
        (447.75389005, b'y', 2, 2),  # LM - LAKY
        (642.38949664, b'b', 1, 1),  # L - KLM
        (713.42661043, b'b', 2, 1),  # LA - KLM
        (781.41643967, b'y', 1, 1),  # M - LAKY
        (841.52157344, b'b', 3, 1),  # LAK - KLM
        (894.50050364, b'y', 2, 1),  # LM - LAKY

        # loss of 1 water
        (182.08116968 - 18.01056027, b'y', 1, 1),  # Y
        (242.18630346 - 18.01056027, b'b', 2, 1),  # KL
        (310.1761327 - 18.01056027, b'y', 2, 1),  # KY
        (381.21324648 - 18.01056027, b'y', 3, 1),  # AKY
        (642.38949664 - 18.01056027, b'b', 1, 1),  # L - KLM
        (713.42661043 - 18.01056027, b'b', 2, 1),  # LA - KLM
        (781.41643967 - 18.01056027, b'y', 1, 1),  # M - LAKY
        (841.52157344 - 18.01056027, b'b', 3, 1),  # LAK - KLM
        (894.50050364 - 18.01056027, b'y', 2, 1),  # LM - LAKY
        (91.54422308 - 9.005280135, b'y', 1, 2),  # Y
        (121.59678996 - 9.005280135, b'b', 2, 2),  # KL
        (155.59170458 - 9.005280135, b'y', 2, 2),  # KY
        (191.11026147 - 9.005280135, b'y', 3, 2),  # AKY
        (321.69838655 - 9.005280135, b'b', 1, 2),  # L - KLM
        (357.21694345 - 9.005280135, b'b', 2, 2),  # LA - KLM
        (391.21185807 - 9.005280135, b'y', 1, 2),  # M - LAKY
        (421.26442495 - 9.005280135, b'b', 3, 2),  # LAK - KLM
        (447.75389005 - 9.005280135, b'y', 2, 2),  # LM - LAKY

        # loss of 2 water
        (642.38949664 - 18.01056027 * 2, b'b', 1, 1),  # L - KLM
        (713.42661043 - 18.01056027 * 2, b'b', 2, 1),  # LA - KLM
        (781.41643967 - 18.01056027 * 2, b'y', 1, 1),  # M - LAKY
        (841.52157344 - 18.01056027 * 2, b'b', 3, 1),  # LAK - KLM
        (894.50050364 - 18.01056027 * 2, b'y', 2, 1),  # LM - LAKY
        (321.69838655 - 9.005280135 * 2, b'b', 1, 2),  # L - KLM
        (357.21694345 - 9.005280135 * 2, b'b', 2, 2),  # LA - KLM
        (391.21185807 - 9.005280135 * 2, b'y', 1, 2),  # M - LAKY
        (421.26442495 - 9.005280135 * 2, b'b', 3, 2),  # LAK - KLM
        (447.75389005 - 9.005280135 * 2, b'y', 2, 2),  # LM - LAKY

        # precursor
        (1021.588181305 + 1.007276466879, b'P', 3, 1),
        (1021.588181305/2 + 1.007276466879, b'P', 3, 2),

        # loss of water
        (1021.588181305 + 1.007276466879 - 18.01056027, b'P', 3, 1),
        (1021.588181305 / 2 + 1.007276466879 - 18.01056027/2, b'P', 3, 2),

        # loss of 2 water
        (1021.588181305 + 1.007276466879 - 18.01056027*2, b'P', 3, 1),
        (1021.588181305 / 2 + 1.007276466879 - 18.01056027, b'P', 3, 2),

    ], dtype=mock_frag_dtype)

    expected_fragments.sort(order=['mz'])

    fragments = fragment_crosslinked_peptide_pair(
        pep1_index=0, pep2_index=1, link_pos1=linkpos1, link_pos2=linkpos2,
        context=ctx, add_precursor=ctx.config.fragmentation.add_precursor,
        crosslinker=ctx.config.crosslinker[0])

    fragments = include_losses(fragments, [0, 1], ctx)

    fragments = spread_charges(fragments, ctx, charge)
    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])


def test_fragment_crosslinked_peptide_pair_with_precursor():
    config = Config(crosslinker=[Crosslinker.BS3], ms2_tol='1ppm',
                    fragmentation=FragmentationConfig(add_precursor=True))
    ctx = MockContext(config)
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))
    linkpos1 = 2
    linkpos2 = 0

    # expected fragments
    # convention: only the first peptide will generate the precursor ions
    add_1st_precursor = ctx.config.fragmentation.add_precursor
    expected_pep1_frags = fragment_crosslinked_peptide(frag_pep_index=0,
                                                       cl_pep_index=1,
                                                       link_pos=linkpos1,
                                                       crosslinker=ctx.config.crosslinker[0],
                                                       context=ctx,
                                                       add_precursor=add_1st_precursor)

    expected_pep2_frags = fragment_crosslinked_peptide(frag_pep_index=1,
                                                       cl_pep_index=0,
                                                       link_pos=linkpos2,
                                                       crosslinker=ctx.config.crosslinker[0],
                                                       context=ctx, add_precursor=False)
    expected_pep2_frags['pep_id'] = 2

    # swap the order of ranges for pep2
    pep2_range0 = expected_pep2_frags['ranges'][:, 0].copy()
    pep2_range1 = expected_pep2_frags['ranges'][:, 1].copy()
    expected_pep2_frags['ranges'][:, 0] = pep2_range1
    expected_pep2_frags['ranges'][:, 1] = pep2_range0
    expected_pep1_frags.sort(order='mz')
    expected_pep2_frags.sort(order='mz')

    # 2nd peptide will not generate precursor fragments

    all_frags = fragment_crosslinked_peptide_pair(pep1_index=0, pep2_index=1,
                                                  link_pos1=linkpos1, link_pos2=linkpos2,
                                                  crosslinker=ctx.config.crosslinker[0],
                                                  context=ctx,
                                                  add_precursor=add_1st_precursor)

    all_frags.sort(order='mz')
    pep1_frags = all_frags[all_frags['pep_id'] == 1]
    pep2_frags = all_frags[all_frags['pep_id'] == 2]

    compare_numpy(expected_pep1_frags, pep1_frags)
    compare_numpy(expected_pep2_frags, pep2_frags)
    assert expected_pep1_frags.size + expected_pep2_frags.size == all_frags.size


def test_fragment_crosslinked_peptide_multisite():

    ctx = MockContext(Config(crosslinker=[Crosslinker.BS3]))
    ctx.setup_peptide_db(np.array([b'KING', b'LAKTYETTLEK']))

    # fragments for LAKTYETTLEK-KING (charge:2)
    # for linkpos1: 2 or 4 on pep1 and linkpos2: 0
    expected_fragments = np.array([
        # mz, ion_type, idx, charge
        (57.54930846, b'b', 1, 2),    # L
        (74.06004032, b'y', 1, 2),    # K
        (93.06786536, b'b', 2, 2),    # LA
        (114.09134046, b'b', 1, 1),    # L
        (138.58133687, b'y', 2, 2),    # EK
        (147.11280418, b'y', 1, 1),    # K
        (157.11534686, b'b', 3, 2),    # LAK
        (185.12845425, b'b', 2, 1),    # LA
        (195.12336886, b'y', 3, 2),    # LEK
        (207.6391861, b'b', 4, 2),    # LAKT
        (245.64720809, b'y', 4, 2),    # TLEK
        (276.15539727, b'y', 2, 1),    # EK
        (296.17104732, b'y', 5, 2),    # TTLEK
        (313.22341726, b'b', 3, 1),    # LAK
        (360.69234387, b'y', 6, 2),    # ETTLEK
        (389.23946125, b'y', 3, 1),    # LEK
        (414.27109573, b'b', 4, 1),    # LAKT
        (441.27637809, b'b', 3, 2),    # LAK + KING
        (442.22400814, b'y', 7, 2),    # YETTLEK
        (490.28713971, b'y', 4, 1),    # TLEK
        (491.80021732, b'b', 4, 2),    # LAKT + KING
        (492.74784737, b'y', 8, 2),    # TYETTLEK
        (573.33188159, b'b', 5, 2),    # LAKTY + KING
        (591.33481818, b'y', 5, 1),    # TTLEK
        (637.85317813, b'b', 6, 2),    # LAKTYE + KING
        (688.37701736, b'b', 7, 2),    # LAKTYET + KING
        (720.37741127, b'y', 6, 1),    # ETTLEK
        (726.38503936, b'y', 7, 2),    # YETTLEK + KING
        (738.9008566, b'b', 8, 2),    # LAKTYETT + KING
        (776.90887859, b'y', 8, 2),    # TYETTLEK + KING
        (795.44288859, b'b', 9, 2),    # LAKTYETTL + KING
        (840.9563601, b'y', 9, 2),    # KTYETTLEK + KING
        (859.96418513, b'b', 10, 2),    # LAKTYETTLE + KING
        (876.47491699, b'y', 10, 2),    # AKTYETTLEK + KING
        (881.54547971, b'b', 3, 1),    # LAK + KING
        (883.4407398, b'y', 7, 1),    # YETTLEK
        (982.59315817, b'b', 4, 1),    # LAKT + KING
        (984.48841827, b'y', 8, 1),    # TYETTLEK
        (1145.65648671, b'b', 5, 1),    # LAKTY + KING
        (1274.69907979, b'b', 6, 1),    # LAKTYE + KING
        (1375.74675826, b'b', 7, 1),    # LAKTYET + KING
        (1451.76280225, b'y', 7, 1),    # YETTLEK + KING
        (1476.79443673, b'b', 8, 1),    # LAKTYETT + KING
        (1552.81048072, b'y', 8, 1),    # TYETTLEK + KING
        (1589.87850071, b'b', 9, 1),    # LAKTYETTL + KING
        (1680.90544373, b'y', 9, 1),    # KTYETTLEK + KING
        (1718.9210938, b'b', 10, 1),    # LAKTYETTLE + KING
        (1751.94255752, b'y', 10, 1),    # AKTYETTLEK + KING
    ], dtype=mock_frag_dtype)

    fragments = fragment_crosslinked_peptide(frag_pep_index=1,
                                             cl_pep_index=0,
                                             link_pos=[2, 4],
                                             crosslinker=ctx.config.crosslinker[0],
                                             context=ctx)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort(order='mz')
    compare_numpy(expected_fragments, fragments, ['mz', 'ion_type', 'idx', 'charge'])


def test_fragment_crosslinked_peptide_pair_multisite():

    ctx = MockContext(Config(crosslinker=[Crosslinker.BS3]))
    ctx.setup_peptide_db(np.array([b'KING', b'LAKTYETTLEK']))

    # fragments for LAKTYETTLEK-KING (charge:2)
    # for linkpos1: 2 or 4 on pep1 and linkpos2: 0
    expected_fragments = np.array([
        # mz, ion_type, idx, charge
        (38.52328860177, b'y', 1, 2),  	# G,
        (57.54930846677, b'b', 1, 2),  	# L,
        (74.06003810177, b'y', 1, 2),  	# K,
        (76.03930073677, b'y', 1, 1),  	# G,
        (93.06786546677, b'b', 2, 2),  	# LA,
        (95.54475210177, b'y', 2, 2),  	# NG,
        (114.09134046676999, b'b', 1, 1),  	# L,
        (138.58133460177, b'y', 2, 2),  	# EK,
        (147.11279973677, b'y', 1, 1),  	# K,
        (152.08678410177, b'y', 3, 2),  	# ING,
        (157.11534696677, b'b', 3, 2),  	# LAK,
        (185.12845446677, b'b', 2, 1),  	# LA,
        (190.08222773677, b'y', 2, 1),  	# NG,
        (195.12336660177, b'y', 3, 2),  	# LEK,
        (207.63918596677, b'b', 4, 2),  	# LAKT,
        (245.64720560177, b'y', 4, 2),  	# TLEK,
        (276.15539273677, b'y', 2, 1),  	# EK,
        (296.17104460177, b'y', 5, 2),  	# TTLEK,
        (303.16629173677, b'y', 3, 1),  	# ING,
        (313.22341746677, b'b', 3, 1),  	# LAK,
        (360.69234110177, b'y', 6, 2),  	# ETTLEK,
        (389.23945673677, b'y', 3, 1),  	# LEK,
        (414.27109546677, b'b', 4, 1),  	# LAKT,
        (441.27637110177, b'b', 3, 2),  	# LAK + KING,
        (442.22400560177, b'y', 7, 2),  	# YETTLEK,
        (490.28713473677, b'y', 4, 1),  	# TLEK,
        (491.80021010177, b'b', 4, 2),  	# LAKT + KING,
        (492.74784460177, b'y', 8, 2),  	# TYETTLEK,
        (573.33187460177, b'b', 5, 2),  	# LAKTY + KING,
        (591.33481273677, b'y', 5, 1),  	# TTLEK,
        (637.85317110177, b'b', 6, 2),  	# LAKTYE + KING,
        (688.37701010177, b'b', 7, 2),  	# LAKTYET + KING,
        (720.37740573677, b'y', 6, 1),  	# ETTLEK,
        (726.38502973677, b'y', 7, 2),  	# YETTLEK + KING,
        (738.90084910177, b'b', 8, 2),  	# LAKTYETT + KING,
        (776.90886873677, b'y', 8, 2),  	# TYETTLEK + KING,
        (781.9374316017701, b'b', 1, 2),  	# K + LAKTYETTLEK,
        (795.44288110177, b'b', 9, 2),  	# LAKTYETTL + KING,
        (838.47946360177, b'b', 2, 2),  	# KI + LAKTYETTLEK,
        (840.95635023677, b'y', 9, 2),  	# KTYETTLEK + KING,
        (859.96417760177, b'b', 10, 2),  	# LAKTYETTLE + KING,
        (876.47490723677, b'y', 10, 2),  	# AKTYETTLEK + KING,
        (881.54546573677, b'b', 3, 1),  	# LAK + KING,
        (883.44073473677, b'y', 7, 1),  	# YETTLEK,
        (895.50092710177, b'b', 3, 2),  	# KIN + LAKTYETTLEK,
        (982.59314373677, b'b', 4, 1),  	# LAKT + KING,
        (984.48841273677, b'y', 8, 1),  	# TYETTLEK,
        (1145.65647273677, b'b', 5, 1),  	# LAKTY + KING,
        (1274.69906573677, b'b', 6, 1),  	# LAKTYE + KING,
        (1375.74674373677, b'b', 7, 1),  	# LAKTYET + KING,
        (1451.76278300677, b'y', 7, 1),  	# YETTLEK + KING,
        (1476.79442173677, b'b', 8, 1),  	# LAKTYETT + KING,
        (1552.81046100677, b'y', 8, 1),  	# TYETTLEK + KING,
        (1562.8675867367701, b'b', 1, 1),  	# K + LAKTYETTLEK,
        (1589.87848573677, b'b', 9, 1),  	# LAKTYETTL + KING,
        (1675.95165073677, b'b', 2, 1),  	# KI + LAKTYETTLEK,
        (1680.90542400677, b'y', 9, 1),  	# KTYETTLEK + KING,
        (1718.92107873677, b'b', 10, 1),  	# LAKTYETTLE + KING,
        (1751.94253800677, b'y', 10, 1),  	# AKTYETTLEK + KING,
        (1789.99457773677, b'b', 3, 1),  	# KIN + LAKTYETTLEK,
    ], dtype=mock_frag_dtype)

    fragments = fragment_crosslinked_peptide_pair(pep1_index=1,
                                                  pep2_index=0,
                                                  link_pos1=[2, 4],
                                                  link_pos2=0,
                                                  crosslinker=ctx.config.crosslinker[0],
                                                  context=ctx)

    fragments = spread_charges(fragments, ctx, 2)
    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])


def test_fragment_linear_peptide():
    ctx = MockContext(Config())

    # setup peptide_db
    peptides = np.array([b'LVTDLTK'])
    ctx.setup_peptide_db(peptides)

    # fragments for LVTDLTK (charge:2)
    # generated with mMass (http://www.mmass.org)
    expected_fragments = np.array([
        57.5493,  # b1  [1-1]   1   .L.v
        74.0600,  # y1	[7-7]	2	t.K.
        107.0835,  # b2	[1-2]	2	.LV.t
        114.0913,  # b1 [1-1]   2   .L.v
        124.5839,  # y2	[6-7]	2	l.TK.
        147.1128,  # y1	[7-7]	1	t.K.
        157.6074,  # b3	[1-3]	2	.LVT.d
        181.1259,  # y3	[5-7]	2	d.LTK.
        213.1598,  # b2	[1-2]	1	.LV.t
        215.1208,  # b4	[1-4]	2	.LVTD.l
        238.6394,  # y4	[4-7]	2	t.DLTK.
        248.1605,  # y2	[6-7]	1	l.TK.
        271.6629,  # b5	[1-5]	2	.LVTDL.t
        289.1632,  # y5	[3-7]	2	v.TDLTK.
        314.2074,  # b3	[1-3]	1	.LVT.d
        322.1867,  # b6	[1-6]	2	.LVTDLT.k
        338.6974,  # y6	[2-7]	2	l.VTDLTK.
        361.2445,  # y3	[5-7]	1	d.LTK.
        429.2344,  # b4	[1-4]	1	.LVTD.l
        476.2715,  # y4	[4-7]	1	t.DLTK.
        542.3184,  # b5	[1-5]	1	.LVTDL.t
        577.3192,  # y5	[3-7]	1	v.TDLTK.
        643.3661,  # b6	[1-6]	1	.LVTDLT.k
        676.3876  # y6	[2-7]	1	l.VTDLTK.
    ])

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = spread_charges(fragments, ctx, 2)
    expected_fragments.sort()

    np.testing.assert_array_almost_equal(expected_fragments, fragments["mz"], decimal=4)

    # add precursor
    precursor_fragments = np.array([
        395.23951,  # P  [0-7]  2   LVTDLTK
        789.47169,  # P  [0-7]  1   LVTDLTK
    ])
    expected_fragments = np.concatenate([expected_fragments, precursor_fragments])
    expected_fragments.sort()

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx,
                                        add_precursor=True)
    fragments = spread_charges(fragments, ctx, 2)

    np.testing.assert_array_almost_equal(expected_fragments, fragments["mz"], decimal=4)


def test_fragment_linear_peptide_etd():

    ctx = MockContext(
        Config(fragmentation=FragmentationConfig(nterm_ions=["c"], cterm_ions=["z"]))
    )

    # setup peptide_db
    peptides = np.array([b'LVTDLTK'])
    ctx.setup_peptide_db(peptides)

    # fragments for LVTDLTK (charge:2)
    # generated with Xi gui
    expected_fragments = np.array([  # mz, sequence, position, charge
        131.11788957177,  # c1  L
        230.18630357177,  # c2  LV
        331.23398157177,  # c3  LVT
        446.26092457177,  # c4  LVTD
        559.34498857177,  # c5  LVTDL
        660.39266657177,  # c6  LVTDLT
        659.36102743177,  # z6  VTDLTK
        560.29261343177,  # z5  TDLTK
        459.24493543177,  # z4  DLTK
        344.21799243177,  # z3  LTK
        231.13392843177,  # z2  TK
        130.08625043177,  # z1  K
    ])
    expected_fragments = expected_fragments.round(4)
    expected_fragments.sort()

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = include_losses(fragments, [0], ctx)
    np.testing.assert_array_almost_equal(expected_fragments, fragments["mz"], decimal=4)

    # add precursor
    precursor_fragments = np.array([
        789.47169,        # P   LVTDLTK
    ])
    expected_fragments = np.concatenate([expected_fragments, precursor_fragments])
    expected_fragments.sort()

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx,
                                        add_precursor=True)
    fragments.sort()
    np.testing.assert_array_almost_equal(expected_fragments, fragments["mz"], decimal=4)


def test_fragment_linear_peptide_ethcd():
    # test generation of b,c and y,z ions
    ctx = MockContext(
        Config(fragmentation=FragmentationConfig(nterm_ions=["b", "c"], cterm_ions=["y", "z"]))
    )

    # setup peptide_db
    peptides = np.array([b'LVTDLTK'])
    ctx.setup_peptide_db(peptides)

    # fragments for LVTDLTK (charge:1)
    # generated with Xi gui
    expected_fragments = [  # mz, sequence, position, charge
        114.09134046677,  # b1  L
        213.15975446677,  # b2  LV
        314.20743246677,  # b3  LVT
        429.23437546677,  # b4  LVTD
        542.31843946677,  # b5  LVTDL
        643.36611746677,  # b6  LVTDLT
        676.38757673677,  # y6  VTDLTK
        577.31916273677,  # y5  TDLTK
        476.27148473677,  # y4  DLTK
        361.24454173677,  # y3  LTK
        248.16047773677,  # y2  TK
        147.11279973677,  # y1  K
        131.11788957177,  # c1  L
        230.18630357177,  # c2  LV
        331.23398157177,  # c3  LVT
        446.26092457177,  # c4  LVTD
        559.34498857177,  # c5  LVTDL
        660.39266657177,  # c6  LVTDLT
        659.36102743177,  # z6  VTDLTK
        560.29261343177,  # z5  TDLTK
        459.24493543177,  # z4  DLTK
        344.21799243177,  # z3  LTK
        231.13392843177,  # z2  TK
        130.08625043177,  # z1  K
    ]
    expected_fragments.sort()

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments.sort()
    np.testing.assert_array_almost_equal(expected_fragments, fragments["mz"], decimal=4)

    # Add precursor
    precursor_fragments = [
        789.47169,        # P   LVTDLTK
    ]

    expected_fragments += precursor_fragments
    expected_fragments.sort()

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx,
                                        add_precursor=True)
    fragments.sort()
    np.testing.assert_array_almost_equal(expected_fragments, fragments["mz"], decimal=4)


def test_fragment_linear_peptide_losses():

    water_loss = Loss(name='H2O', composition="H2O1", specificity=['S', 'T', 'E', 'D'])

    ctx = MockContext(
        Config(fragmentation=FragmentationConfig(losses=[water_loss], max_nloss=2))
    )

    # setup peptide_db
    peptides = np.array([b'LVTDLTK'])
    ctx.setup_peptide_db(peptides)

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = include_losses(fragments, [0], ctx)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()

    # fragments for LVTDLTK (charge:2), and water losses
    # generated with Xi (fragment gui)
    expected_fragments = np.array([
        (114.09134046677, 1, True, b'', 0, b'', b'b', b'n', 1, 1, [[0, 1], [0, 0]]),  # L
        (57.54930846677, 2, True, b'', 0, b'', b'b', b'n', 1, 1, [[0, 1], [0, 0]]),  # L
        (213.15975446677, 1, True, b'', 0, b'', b'b', b'n', 2, 1, [[0, 2], [0, 0]]),  # LV
        (107.08351546677, 2, True, b'', 0, b'', b'b', b'n', 2, 1, [[0, 2], [0, 0]]),  # LV
        (314.20743246677, 1, True, b'', 0, b'', b'b', b'n', 3, 1, [[0, 3], [0, 0]]),  # LVT
        (157.60735446677, 2, True, b'', 0, b'', b'b', b'n', 3, 1, [[0, 3], [0, 0]]),  # LVT
        (429.23437546677, 1, True, b'', 0, b'', b'b', b'n', 4, 1, [[0, 4], [0, 0]]),  # LVTD
        (215.12082596677, 2, True, b'', 0, b'', b'b', b'n', 4, 1, [[0, 4], [0, 0]]),  # LVTD
        (542.31843946677, 1, True, b'', 0, b'', b'b', b'n', 5, 1, [[0, 5], [0, 0]]),  # LVTDL
        (271.66285796677, 2, True, b'', 0, b'', b'b', b'n', 5, 1, [[0, 5], [0, 0]]),  # LVTDL
        (643.36611746677, 1, True, b'', 0, b'', b'b', b'n', 6, 1, [[0, 6], [0, 0]]),  # LVTDLT
        (322.18669696677, 2, True, b'', 0, b'', b'b', b'n', 6, 1, [[0, 6], [0, 0]]),  # LVTDLT
        (676.38757673677, 1, True, b'', 0, b'', b'y', b'c', 6, 1, [[1, 7], [0, 0]]),  # VTDLTK
        (338.69742660177, 2, True, b'', 0, b'', b'y', b'c', 6, 1, [[1, 7], [0, 0]]),  # VTDLTK
        (577.31916273677, 1, True, b'', 0, b'', b'y', b'c', 5, 1, [[2, 7], [0, 0]]),  # TDLTK
        (289.16321960177, 2, True, b'', 0, b'', b'y', b'c', 5, 1, [[2, 7], [0, 0]]),  # TDLTK
        (476.27148473677, 1, True, b'', 0, b'', b'y', b'c', 4, 1, [[3, 7], [0, 0]]),  # DLTK
        (238.63938060177, 2, True, b'', 0, b'', b'y', b'c', 4, 1, [[3, 7], [0, 0]]),  # DLTK
        (361.24454173677, 1, True, b'', 0, b'', b'y', b'c', 3, 1, [[4, 7], [0, 0]]),  # LTK
        (181.12590910177, 2, True, b'', 0, b'', b'y', b'c', 3, 1, [[4, 7], [0, 0]]),  # LTK
        (248.16047773677, 1, True, b'', 0, b'', b'y', b'c', 2, 1, [[5, 7], [0, 0]]),  # TK
        (124.58387710177, 2, True, b'', 0, b'', b'y', b'c', 2, 1, [[5, 7], [0, 0]]),  # TK
        (147.11279973677, 1, True, b'', 0, b'', b'y', b'c', 1, 1, [[6, 7], [0, 0]]),  # K
        (74.06003810177, 2, True, b'', 0, b'', b'y', b'c', 1, 1, [[6, 7], [0, 0]]),  # K

        # loss
        # m/z, ion, charge, sequence
        (115.57859696677, 2, True, b'H2Ox1', 1, b'', b'y', b'c', 2, 1, [[5, 7], [0, 0]]),  # TK
        (148.60207433177, 2, True, b'H2Ox1', 1, b'', b'b', b'n', 3, 1, [[0, 3], [0, 0]]),  # LVT
        (172.12062896677, 2, True, b'H2Ox1', 1, b'', b'y', b'c', 3, 1, [[4, 7], [0, 0]]),  # LTK
        (206.11554583177, 2, True, b'H2Ox1', 1, b'', b'b', b'n', 4, 1, [[0, 4], [0, 0]]),  # LVTD
        (229.63410046677, 2, True, b'H2Ox1', 1, b'', b'y', b'c', 4, 1, [[3, 7], [0, 0]]),  # DLTK
        (230.14991746677, 1, True, b'H2Ox1', 1, b'', b'y', b'c', 2, 1, [[5, 7], [0, 0]]),  # TK
        (262.65757783177, 2, True, b'H2Ox1', 1, b'', b'b', b'n', 5, 1, [[0, 5], [0, 0]]),  # LVTDL
        (280.15793946677, 2, True, b'H2Ox1', 1, b'', b'y', b'c', 5, 1, [[2, 7], [0, 0]]),  # TDLTK
        (296.19687219677, 1, True, b'H2Ox1', 1, b'', b'b', b'n', 3, 1, [[0, 3], [0, 0]]),  # LVT
        (313.18141683177, 2, True, b'H2Ox1', 1, b'', b'b', b'n', 6, 1, [[0, 6], [0, 0]]),  # LVTDLT
        (329.69214646677, 2, True, b'H2Ox1', 1, b'', b'y', b'c', 6, 1, [[1, 7], [0, 0]]),  # VTDLTK
        (343.23398146677, 1, True, b'H2Ox1', 1, b'', b'y', b'c', 3, 1, [[4, 7], [0, 0]]),  # LTK
        (411.22381519677, 1, True, b'H2Ox1', 1, b'', b'b', b'n', 4, 1, [[0, 4], [0, 0]]),  # LVTD
        (458.26092446677, 1, True, b'H2Ox1', 1, b'', b'y', b'c', 4, 1, [[3, 7], [0, 0]]),  # DLTK
        (524.30787919677, 1, True, b'H2Ox1', 1, b'', b'b', b'n', 5, 1, [[0, 5], [0, 0]]),  # LVTDL
        (559.30860246677, 1, True, b'H2Ox1', 1, b'', b'y', b'c', 5, 1, [[2, 7], [0, 0]]),  # TDLTK
        (625.35555719677, 1, True, b'H2Ox1', 1, b'', b'b', b'n', 6, 1, [[0, 6], [0, 0]]),  # LVTDLT
        (658.37701646677, 1, True, b'H2Ox1', 1, b'', b'y', b'c', 6, 1, [[1, 7], [0, 0]]),  # VTDLTK
        (197.11026569677, 2, True, b'H2Ox2', 2, b'', b'b', b'n', 4, 1, [[0, 4], [0, 0]]),  # LVTD
        (220.62882033177, 2, True, b'H2Ox2', 2, b'', b'y', b'c', 4, 1, [[3, 7], [0, 0]]),  # DLTK
        (253.65229769677, 2, True, b'H2Ox2', 2, b'', b'b', b'n', 5, 1, [[0, 5], [0, 0]]),  # LVTDL
        (271.15265933177, 2, True, b'H2Ox2', 2, b'', b'y', b'c', 5, 1, [[2, 7], [0, 0]]),  # TDLTK
        (304.17613669677, 2, True, b'H2Ox2', 2, b'', b'b', b'n', 6, 1, [[0, 6], [0, 0]]),  # LVTDLT
        (320.68686633177, 2, True, b'H2Ox2', 2, b'', b'y', b'c', 6, 1, [[1, 7], [0, 0]]),  # VTDLTK
        (393.21325492677, 1, True, b'H2Ox2', 2, b'', b'b', b'n', 4, 1, [[0, 4], [0, 0]]),  # LVTD
        (440.25036419677, 1, True, b'H2Ox2', 2, b'', b'y', b'c', 4, 1, [[3, 7], [0, 0]]),  # DLTK
        (506.29731892677, 1, True, b'H2Ox2', 2, b'', b'b', b'n', 5, 1, [[0, 5], [0, 0]]),  # LVTDL
        (541.29804219677, 1, True, b'H2Ox2', 2, b'', b'y', b'c', 5, 1, [[2, 7], [0, 0]]),  # TDLTK
        (607.34499692677, 1, True, b'H2Ox2', 2, b'', b'b', b'n', 6, 1, [[0, 6], [0, 0]]),  # LVTDLT
        (640.36645619677, 1, True, b'H2Ox2', 2, b'', b'y', b'c', 6, 1, [[1, 7], [0, 0]]),  # VTDLTK
    ], dtype=fragments.dtype)

    expected_fragments.sort(order='mz')

    np.testing.assert_array_almost_equal(expected_fragments['mz'], fragments['mz'], decimal=4)
    identical_cols = ['charge', 'LN', 'loss', 'nlosses', 'stub', 'ion_type', 'term', 'idx',
                      'pep_id', 'ranges']
    np.testing.assert_array_equal(expected_fragments[identical_cols], fragments[identical_cols])

    # add precursor
    precursor_fragments = np.array([
        (789.47163115047, 1, True, b'', 0, b'', b'P', b'P', 7, 1, [[0, 7], [0, 0]]),
        (395.23945380862, 2, True, b'', 0, b'', b'P', b'P', 7, 1, [[0, 7], [0, 0]]),
        (771.46107088047, 1, True, b'H2Ox1', 1, b'', b'P', b'P', 7, 1, [[0, 7], [0, 0]]),
        (386.23417367362, 2, True, b'H2Ox1', 1, b'', b'P', b'P', 7, 1, [[0, 7], [0, 0]]),
        (753.45051061047, 1, True, b'H2Ox2', 2, b'', b'P', b'P', 7, 1, [[0, 7], [0, 0]]),
        (377.22889353862, 2, True, b'H2Ox2', 2, b'', b'P', b'P', 7, 1, [[0, 7], [0, 0]]),
    ], dtype=fragments.dtype)

    expected_fragments = np.hstack([expected_fragments, precursor_fragments])
    expected_fragments.sort(order='mz')

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx,
                                        add_precursor=True)
    fragments = include_losses(fragments, [0], ctx)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()

    np.testing.assert_array_almost_equal(expected_fragments['mz'], fragments['mz'], decimal=4)
    np.testing.assert_array_equal(expected_fragments[identical_cols], fragments[identical_cols])


def test_fragment_linear_peptide_losses_cterm():

    water_loss = Loss(name='H2O', composition="H2O1", specificity=['S', 'T', 'E', 'D', 'cterm'])

    ctx = MockContext(
        Config(fragmentation=FragmentationConfig(losses=[water_loss],
                                                 max_nloss=4))
    )

    # setup peptide_db
    peptides = np.array([b'LVTDLTK'])
    ctx.setup_peptide_db(peptides)

    # fragments for LVTDLTK (charge:1), and water losses
    # generated with Xi (fragment gui)
    expected_fragments = np.array([
        114.09134046677,  # b1  L
        213.15975446677,  # b2  LV
        314.20743246677,  # b3  LVT
        429.23437546677,  # b4  LVTD
        542.31843946677,  # b5  LVTDL
        643.36611746677,  # b6  LVTDLT
        676.38757673677,  # y6  VTDLTK
        577.31916273677,  # y5  TDLTK
        476.27148473677,  # y4  DLTK
        361.24454173677,  # y3  LTK
        248.16047773677,  # y2  TK
        147.11279973677,  # y1  K
        296.19687219677,  # b3_H20x1  LVT
        411.22381519677,  # b4_H20x1  LVTD
        393.21325492677,  # b4_H20x2  LVTD
        524.30787919677,  # b5_H20x1  LVTDL
        506.29731892677,  # b5_H20x2  LVTDL
        625.35555719677,  # b6_H20x1  LVTDLT
        607.34499692677,  # b6_H20x2  LVTDLT
        589.33443665677,  # b6_H20x3  LVTDLT
        658.37701646677,  # y6_H20x1  VTDLTK
        640.36645619677,  # y6_H20x2  VTDLTK
        622.35589592677,  # y6_H20x3  VTDLTK
        604.34533565677,  # y6_H20x4  VTDLTK
        559.30860246677,  # y5_H20x1  TDLTK
        541.29804219677,  # y5_H20x2  TDLTK
        523.28748192677,  # y5_H20x3  TDLTK
        505.27692165677,  # y5_H20x4  TDLTK
        458.26092446677,  # y4_H20x1  DLTK
        440.25036419677,  # y4_H20x2  DLTK
        422.23980392677,  # y4_H20x3  DLTK
        343.23398146677,  # y3_H20x1  LTK
        325.22342119677,  # y3_H20x2  LTK
        230.14991746677,  # y2_H20x1  TK
        212.13935719677,  # y2_H20x2  TK
        129.10223946677,  # y1_H20x1  K
    ])
    expected_fragments.sort()

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = include_losses(fragments, [0], ctx)
    fragments.sort()

    np.testing.assert_array_almost_equal(expected_fragments, fragments['mz'], decimal=4)


def test_fragment_linear_peptide_multiple_losses():
    water_loss = Loss(name='H2O', composition="H2O1", specificity=['S', 'T', 'E', 'D', 'cterm'])
    ammonia_loss = Loss(name='NH3', composition="N1H3", specificity=['R', 'K', 'N', 'Q', 'nterm'])

    ctx = MockContext(
        Config(fragmentation=FragmentationConfig(losses=[water_loss, ammonia_loss],
                                                 max_nloss=4))
    )

    # setup peptide_db
    peptides = np.array([b'LVTDLTK'])
    ctx.setup_peptide_db(peptides)

    # fragments for LVTDLTK (charge:1), and water losses
    # generated with Xi (fragment gui)
    expected_fragments = np.array([
        # m/z,  Ion, Charge, Sequence
        114.09134046677,  # b1, z1, L
        213.15975446677,  # b2, z1, LV
        314.20743246677,  # b3, z1, LVT
        429.23437546677,  # b4, z1, LVTD
        542.31843946677,  # b5, z1, LVTDL
        643.36611746677,  # b6, z1, LVTDLT
        789.47164073677,  # P, z1, LVTDLTK
        147.11279973677,  # y1, z1, K
        248.16047773677,  # y2, z1, TK
        361.24454173677,  # y3, z1, LTK
        476.27148473677,  # y4, z1, DLTK
        577.31916273677,  # y5, z1, TDLTK
        676.38757673677,  # y6, z1, VTDLTK
        97.06479553677,  # b1_NH3, z1, L
        196.13320953677,  # b2_NH3, z1, LV
        297.18088753677,  # b3_NH3, z1, LVT
        412.20783053677,  # b4_NH3, z1, LVTD
        525.29189453677,  # b5_NH3, z1, LVTDL
        626.33957253677,  # b6_NH3, z1, LVTDLT
        772.44509580677,  # P_NH3, z1, LVTDLTK
        130.08625480677,  # y1_NH3, z1, K
        231.13393280677,  # y2_NH3, z1, TK
        344.21799680677,  # y3_NH3, z1, LTK
        459.24493980677,  # y4_NH3, z1, DLTK
        560.29261780677,  # y5_NH3, z1, TDLTK
        659.36103180677,  # y6_NH3, z1, VTDLTK
        755.41855087677,  # P_NH3x2, z1, LVTDLTK
        296.19687219677,  # b3 H2O, z1, LVT
        411.22381519677,  # b4 H2O, z1, LVTD
        524.30787919677,  # b5 H2O, z1, LVTDL
        625.35555719677,  # b6 H2O, z1, LVTDLT
        771.46108046677,  # P H2O, z1, LVTDLTK
        129.10223946677,  # y1 H2O, z1, K
        230.14991746677,  # y2 H2O, z1, TK
        343.23398146677,  # y3 H2O, z1, LTK
        458.26092446677,  # y4 H2O, z1, DLTK
        559.30860246677,  # y5 H2O, z1, TDLTK
        658.37701646677,  # y6 H2O, z1, VTDLTK
        279.17032726677,  # b3 H2O_NH3, z1, LVT
        394.19727026677,  # b4 H2O_NH3, z1, LVTD
        507.28133426677,  # b5 H2O_NH3, z1, LVTDL
        608.32901226677,  # b6 H2O_NH3, z1, LVTDLT
        754.43453553677,  # P H2O_NH3, z1, LVTDLTK
        112.07569453677,  # y1 H2O_NH3, z1, K
        213.12337253677,  # y2 H2O_NH3, z1, TK
        326.20743653677,  # y3 H2O_NH3, z1, LTK
        441.23437953677,  # y4 H2O_NH3, z1, DLTK
        542.28205753677,  # y5 H2O_NH3, z1, TDLTK
        641.35047153677,  # y6 H2O_NH3, z1, VTDLTK
        737.40799060677,  # P H2O_NH3x2, z1, LVTDLTK
        393.21325492677,  # b4 H2Ox2, z1, LVTD
        506.29731892677,  # b5 H2Ox2, z1, LVTDL
        607.34499692677,  # b6 H2Ox2, z1, LVTDLT
        753.45052019677,  # P H2Ox2, z1, LVTDLTK
        212.13935719677,  # y2 H2Ox2, z1, TK
        325.22342119677,  # y3 H2Ox2, z1, LTK
        440.25036419677,  # y4 H2Ox2, z1, DLTK
        541.29804219677,  # y5 H2Ox2, z1, TDLTK
        640.36645619677,  # y6 H2Ox2, z1, VTDLTK
        376.18670999677,  # b4 H2Ox2_NH3, z1, LVTD
        489.27077399677,  # b5 H2Ox2_NH3, z1, LVTDL
        590.31845199677,  # b6 H2Ox2_NH3, z1, LVTDLT
        736.42397526677,  # P H2Ox2_NH3, z1, LVTDLTK
        195.11281226677,  # y2 H2Ox2_NH3, z1, TK
        308.19687626677,  # y3 H2Ox2_NH3, z1, LTK
        423.22381926677,  # y4 H2Ox2_NH3, z1, DLTK
        524.27149726677,  # y5 H2Ox2_NH3, z1, TDLTK
        623.33991126677,  # y6 H2Ox2_NH3, z1, VTDLTK
        719.39743033677,  # P H2Ox2_NH3x2, z1, LVTDLTK
        589.33443665677,  # b6 H2Ox3, z1, LVTDLT
        735.43995992677,  # P H2Ox3, z1, LVTDLTK
        422.23980392677,  # y4 H2Ox3, z1, DLTK
        523.28748192677,  # y5 H2Ox3, z1, TDLTK
        622.35589592677,  # y6 H2Ox3, z1, VTDLTK
        572.30789172677,  # b6 H2Ox3_NH3, z1, LVTDLT
        718.41341499677,  # P H2Ox3_NH3, z1, LVTDLTK
        405.21325899677,  # y4 H2Ox3_NH3, z1, DLTK
        506.26093699677,  # y5 H2Ox3_NH3, z1, TDLTK
        605.32935099677,  # y6 H2Ox3_NH3, z1, VTDLTK
        717.42939965677,  # P 4xH2O, z1, LVTDLTK
        505.27692165677,  # y5 4xH2O, z1, TDLTK
        604.34533565677,  # y6 4xH2O, z1, VTDLTK
    ])
    expected_fragments.sort()

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx,
                                        add_precursor=True)
    fragments = include_losses(fragments, [0], ctx)
    fragments.sort()

    np.testing.assert_array_almost_equal(expected_fragments, fragments['mz'], decimal=4)


def test_fragment_modified_linear_peptide():

    test_mod = Modification(name='cm', specificity=['C'], type='fixed', composition="C2H3O1N1")
    ctx = MockContext(
        Config(modification=ModificationConfig(modifications=[test_mod]))
    )

    # setup peptide_db
    peptides = np.array([b'LAcmCK'])
    ctx.setup_peptide_db(peptides)

    # fragments for LACK (charge:2)
    # with carbamidomethylation modification on C
    # generated with mMass (http://www.mmass.org)
    expected_fragments = [
        57.5493,  # b1              1
        74.0600,  # y1	[4-4]		2	c.K.
        93.0679,  # b2	[1-2]		2	.LA.c
        114.0913,  # b1             2
        147.1128,  # y1	[4-4]		1	c.K.
        154.0754,  # y2	[3-4]		2	a.CK. [1xCarbamidomethyl]
        173.0832,  # b3	[1-3]		2	.LAC.k [1xCarbamidomethyl]
        185.1285,  # b2	[1-2]		1	.LA.c
        189.5939,  # y3	[2-4]		2	l.ACK. [1xCarbamidomethyl]
        307.1435,  # y2	[3-4]		1	a.CK. [1xCarbamidomethyl]
        345.1591,  # b3	[1-3]		1	.LAC.k [1xCarbamidomethyl]
        378.1806,  # y3	[2-4]		1	l.ACK. [1xCarbamidomethyl]
    ]

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = spread_charges(fragments, ctx, 2)

    np.testing.assert_array_almost_equal(fragments["mz"], expected_fragments, decimal=4)


def test_fragment_modified_linear_peptide_with_losses():
    """ features a peptide with same unmodified base sequence from c and n terminal up to b/y 2 """

    # ToDo: cmM specificity in loss because of bug with just nterm (see Xi-205)
    test_loss = Loss(name='H2O', mass=1, specificity=['cmM', 'nterm'])
    test_mod = Modification(name='cm', specificity=['C'], type='fixed', composition="C2H3O1N1")
    config = Config(fragmentation=FragmentationConfig(losses=[test_loss],
                                                      max_nloss=3),
                    modification=ModificationConfig(modifications=[test_mod]))
    ctx = MockContext(config)

    # setup peptide_db
    peptides = np.array([b'CKLAcmCK'])
    ctx.setup_peptide_db(peptides)

    # fragments for CKLAcmCK(charge:1)
    # with carbamidomethylation modification on C4
    # generated with xi (PeptideToIon.bat)
    expected_fragments = [
        104.01646046677,  # b1, C, 103.009184,
        232.11142346677002,  # b2, CK, 231.104147,
        345.19548746677003,  # b3, CKL, 344.188211,
        416.23260146677,  # b4, CKLA, 415.225325,
        576.26325146677,  # b5, CKLACcm, 575.255975,
        619.3595907367701,  # y5, KLACcmK, 618.3523142700001,
        491.26462773677,  # y4, LACcmK, 490.25735127,
        378.18056373677,  # y3, ACcmK, 377.17328727,
        307.14344973677004,  # y2, CcmK, 306.13617327000003,
        147.11279973677,  # y1, K, 146.10552327,
        # 722.3687747367701,  # P, CKLACcmK, 721.3614982700001,
        103.01646046677,  # b1_testx1, C, 102.009184,
        231.11142346677002,  # b2_testx1, CK, 230.104147,
        344.19548746677003,  # b3_testx1, CKL, 343.188211,
        415.23260146677,  # b4_testx1, CKLA, 414.225325,
        575.26325146677,  # b5_testx1, CKLACcm, 574.255975,
    ]

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = include_losses(fragments, [0], ctx)
    fragments.sort()

    np.testing.assert_array_almost_equal(fragments["mz"], sorted(expected_fragments), decimal=4)


def test_fragment_modified_linear_peptide_with_losses2():
    """ features a peptide with same modified base sequence from c and n terminal up to b/y 2 """

    # ToDo: cmM specificity in loss because of bug with just nterm (see Xi-205)
    test_loss = Loss(name='H2O', mass=1, specificity=['cmM', 'nterm'])
    test_mod = Modification(name='cm', specificity=['C'], type='fixed', composition="C2H3O1N1")
    config = Config(fragmentation=FragmentationConfig(losses=[test_loss],
                                                      max_nloss=3),
                    modification=ModificationConfig(modifications=[test_mod]))
    ctx = MockContext(config)

    # setup peptide_db
    peptides = np.array([b'cmCKLAcmCK'])
    ctx.setup_peptide_db(peptides)

    # fragments for cmCKLAcmCK(charge:1)
    # with carbamidomethylation modification on C0 and C4
    # generated with xi (PeptideToIon.bat)
    expected_fragments = [
        161.03792646677002,  # b1, Ccm, 160.03065,
        289.13288946677005,  # b2, CcmK, 288.12561300000004,
        402.21695346677006,  # b3, CcmKL, 401.20967700000006,
        473.25406746677004,  # b4, CcmKLA, 472.24679100000003,
        633.2847174667701,  # b5, CcmKLACcm, 632.2774410000001,
        619.3595907367701,  # y5, KLACcmK, 618.3523142700001,
        491.26462773677,  # y4, LACcmK, 490.25735127,
        378.18056373677,  # y3, ACcmK, 377.17328727,
        307.14344973677004,  # y2, CcmK, 306.13617327000003,
        147.11279973677,  # y1, K, 146.10552327,
        # 779.3902407367701,  # P, CcmKLACcmK, 778.3829642700001,
        160.03792646677002,  # b1_testx1, Ccm, 159.03065,
        288.13288946677005,  # b2_testx1, CcmK, 287.12561300000004,
        401.21695346677006,  # b3_testx1, CcmKL, 400.20967700000006,
        472.25406746677004,  # b4_testx1, CcmKLA, 471.24679100000003,
        632.2847174667701,  # b5_testx1, CcmKLACcm, 631.2774410000001,
    ]

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = include_losses(fragments, [0], ctx)
    fragments.sort()

    np.testing.assert_array_almost_equal(fragments["mz"], sorted(expected_fragments), decimal=4)


def test_fragment_modified_linear_peptide_with_losses_modified_aa_specificity():
    """features a loss with modified AA specificity"""

    test_loss = Loss(name='CH3SOH', mass=63.99828547, specificity=['oxM'])
    test_mod = Modification(name='ox', specificity=['M'], type='variable', composition="O1")
    config = Config(fragmentation=FragmentationConfig(losses=[test_loss],
                                                      max_nloss=1),
                    modification=ModificationConfig(modifications=[test_mod]))
    ctx = MockContext(config)

    # setup peptide_db
    peptides = np.array([b'oxMFR'])
    ctx.setup_peptide_db(peptides)

    # fragments for oxMFR (charge:2)
    # with oxidation modification on M0
    # generated with xi (PeptideToIon.bat)
    expected_fragments = np.array([
        # mz, charge, LN, loss, nlosses, stub, ion_type, term, idx, pep_id, ranges
        (42.52583123177, 2, True, 'CH3SOHx1', 1, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # oxM
        (74.52497396677, 2, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # oxM
        (84.04438599677, 1, True, 'CH3SOHx1', 1, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # oxM
        (88.06311210177, 2, True, b'', 0, b'', b'y', b'c', 1, 1, ((2, 3), (0, 0))),  # R
        (116.06003823177, 2, True, 'CH3SOHx1', 1, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # oxMF
        (148.04267146677, 1, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # oxM
        (148.05918096677, 2, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # oxMF
        (161.59731910177, 2, True, b'', 0, b'', b'y', b'c', 2, 1, ((1, 3), (0, 0))),  # FR
        (175.11894773677, 1, True, b'', 0, b'', b'y', b'c', 1, 1, ((2, 3), (0, 0))),  # R
        (203.11587831, 2, True, 'CH3SOHx1', 1, b'', b'P', b'P', 3, 1, ((0, 3), (0, 0))),  # oxMFR
        (231.11279999677, 1, True, 'CH3SOHx1', 1, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # oxMF
        (235.11501660177, 2, True, b'', 0, b'', b'P', b'P', 3, 1, ((0, 3), (0, 0))),  # oxMFR
        (295.11108546677, 1, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # oxMF
        (322.18736173677, 1, True, b'', 0, b'', b'y', b'c', 2, 1, ((1, 3), (0, 0))),  # FR
        (405.22448015, 1, True, 'CH3SOHx1', 1, b'', b'P', b'P', 3, 1, ((0, 3), (0, 0))),  # oxMFR
        (469.22275673677, 1, True, b'', 0, b'', b'P', b'P', 3, 1, ((0, 3), (0, 0))),  # oxMFR
    ], dtype=dtypes.fragments)

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx, add_precursor=True)
    fragments = include_losses(fragments, [0], ctx)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()

    for name in fragments.dtype.names:
        if np.issubdtype(fragments.dtype[name], np.number):
            np.testing.assert_allclose(
                fragments[name], expected_fragments[name], rtol=2e-7,
                err_msg=f'{name} doesnae match, '
                f'expected_fragments {expected_fragments[name]}, got {fragments[name]}')
        else:
            assert np.all(fragments[name] == expected_fragments[name]), f'{name} doesnae match, ' \
                f'expected_fragments {expected_fragments[name]}, got {fragments[name]}'


def test_fragment_linear_peptide_with_non_standard_aas():
    """features the non-standard amino acids U (selenocysteine) and O (pyrrolysine)."""

    mod1 = Modification(name='um', specificity=['U'], type='variable', composition="H1")
    mod2 = Modification(name='om', specificity=['O'], type='variable', composition="H1")
    config = Config(modification=ModificationConfig(modifications=[mod1, mod2]))
    ctx = MockContext(config)

    # setup peptide_db
    peptides = np.array([b'RULOR'])
    ctx.setup_peptide_db(peptides)

    # fragments for RULOR (charge:2)
    # generated with LF's spreadsheet (FragmentWeights.xls)
    expected_fragments = np.array([
        # mz, charge, LN, loss, nlosses, stub, ion_type, term, idx, pep_id, ranges
        (79.057831991879, 2, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # R
        (88.063112126879, 2, True, b'', 0, b'', b'y', b'c', 1, 1, ((4, 5), (0, 0))),  # R
        (154.534648694379, 2, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # RU
        (157.108387516879, 1, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # R
        (175.118947786879, 1, True, b'', 0, b'', b'y', b'c', 1, 1, ((4, 5), (0, 0))),  # R
        (206.636975589379, 2, True, b'', 0, b'', b'y', b'c', 2, 1, ((3, 5), (0, 0))),  # OR
        (211.076680701879, 2, True, b'', 0, b'', b'b', b'n', 3, 1, ((0, 3), (0, 0))),  # RUL
        (263.179007596879, 2, True, b'', 0, b'', b'y', b'c', 3, 1, ((2, 5), (0, 0))),  # LOR
        (308.062020921879, 1, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # RU
        (329.650544164379, 2, True, b'', 0, b'', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # RULO
        (338.655824299379, 2, True, b'', 0, b'', b'y', b'c', 4, 1, ((1, 5), (0, 0))),  # ULOR
        (412.266674711879, 1, True, b'', 0, b'', b'y', b'c', 2, 1, ((3, 5), (0, 0))),  # OR
        (416.706379824379, 2, True, b'', 0, b'', b'P', b'P', 5, 1, ((0, 5), (0, 0))),  # RULOR
        (421.146084936879, 1, True, b'', 0, b'', b'b', b'n', 3, 1, ((0, 3), (0, 0))),  # RUL
        (525.350738726879, 1, True, b'', 0, b'', b'y', b'c', 3, 1, ((2, 5), (0, 0))),  # LOR
        (658.293811861879, 1, True, b'', 0, b'', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # RULO
        (676.304372131879, 1, True, b'', 0, b'', b'y', b'c', 4, 1, ((1, 5), (0, 0))),  # ULOR
        (832.405483181879, 1, True, b'', 0, b'', b'P', b'P', 5, 1, ((0, 5), (0, 0))),  # RULOR
    ], dtype=dtypes.fragments)

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx, add_precursor=True)
    fragments = spread_charges(fragments, ctx, 2)

    for name in fragments.dtype.names:
        if np.issubdtype(fragments.dtype[name], np.number):
            np.testing.assert_allclose(
                fragments[name], expected_fragments[name], rtol=2e-7,
                err_msg=f'{name} doesnae match, '
                f'expected_fragments {expected_fragments[name]}, got {fragments[name]}')
        else:
            assert np.all(fragments[name] == expected_fragments[name]), f'{name} doesnae match, ' \
                f'expected_fragments {expected_fragments[name]}, got {fragments[name]}'


def test_fragment_cn_term_modified_linear_peptide():

    n_term_mod = Modification(name='acetyl-', specificity=['X'], type='fixed', composition="C2H2O1")
    c_term_mod = Modification(name='-amide', specificity=['X'], type='fixed', composition="N1H1O-1")

    ctx = MockContext(
        Config(modification=ModificationConfig(modifications=[n_term_mod, c_term_mod]))
    )

    # setup peptide_db
    peptides = np.array([b'acetyl-KLE-amide'])
    ctx.setup_peptide_db(peptides)

    # fragments for KLE(charge:2)
    # with n-terminal acetyl and c-terminal amide
    # generated with mMass (http://www.mmass.org)
    expected_fragments = [
        74.0418,  # y1	[3-3]	2	l.E. [1xAmide]
        86.0601,  # b1	[1-1]	1	.K.e [1xAcetyl]
        130.5839,  # y2	[2-3]	2	k.LE. [1xAmide]
        142.6021,  # b2	[1-2]	2	.KL.e [1xAcetyl]
        147.0764,  # y1	[3-3]	1	l.E. [1xAmide]
        171.1129,  # b1	[1-1]	2	.K.e [1xAcetyl]
        260.1605,  # y2	[2-3]	1	k.LE. [1xAmide]
        284.1969,  # b2	[1-2]	1	.KL.e [1xAcetyl]
    ]

    fragments = fragment_linear_peptide(peptide_index=0, context=ctx)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()

    np.testing.assert_array_almost_equal(fragments['mz'], expected_fragments, decimal=4)

    # Add precursor
    config = Config(modification=ModificationConfig(modifications=[n_term_mod, c_term_mod]),
                    fragmentation=FragmentationConfig(add_precursor=True))
    ctx = MockContext(config)
    ctx.setup_peptide_db(peptides)

    precursor_fragments = [
        430.26600573677,  # precursor z1
        215.63664110177,  # precursor z2
    ]

    expected_fragments += precursor_fragments
    expected_fragments.sort()
    fragments = fragment_linear_peptide(peptide_index=0, context=ctx,
                                        add_precursor=ctx.config.fragmentation.add_precursor)
    fragments = spread_charges(fragments, ctx, 2)

    np.testing.assert_array_almost_equal(fragments['mz'], expected_fragments, decimal=4)


def test_fragment_crosslinked_peptide():

    ctx = MockContext(Config(crosslinker='BS3'))
    # setup peptide_db
    peptides = np.array([b'KLM', b'LAKTYETTLEK'])
    ctx.setup_peptide_db(peptides)

    # fragments for LAKTYETTLEK (charge:2)
    # with KLM as second peptide linked on K2
    # generated Xi1 PeptideToIon.bat
    expected_fragments = np.array([
        # mz, ion_type, idx, charge
        (57.54930846677, b'b', 1, 2),  	# L,
        (74.06003810177, b'y', 1, 2),  	# K,
        (93.06786546677, b'b', 2, 2),  	# LA,
        (114.09134046677, b'b', 1, 1),  	# L,
        (138.58133460177, b'y', 2, 2),  	# EK,
        (147.11279973677, b'y', 1, 1),  	# K,
        (185.12845446677, b'b', 2, 1),  	# LA,
        (195.12336660177, b'y', 3, 2),  	# LEK,
        (245.64720560177, b'y', 4, 2),  	# TLEK,
        (276.15539273677, b'y', 2, 1),  	# EK,
        (296.17104460177, b'y', 5, 2),  	# TTLEK,
        (360.69234110177, b'y', 6, 2),  	# ETTLEK,
        (389.23945673677, b'y', 3, 1),  	# LEK,
        (421.26441810177, b'b', 3, 2),  	# LAK + KLM,
        (442.22400560177, b'y', 7, 2),  	# YETTLEK,
        (471.78825710177, b'b', 4, 2),  	# LAKT + KLM,
        (490.28713473677, b'y', 4, 1),  	# TLEK,
        (492.74784460177, b'y', 8, 2),  	# TYETTLEK,
        (553.31992160177, b'b', 5, 2),  	# LAKTY + KLM,
        (591.33481273677, b'y', 5, 1),  	# TTLEK,
        (617.84121810177, b'b', 6, 2),  	# LAKTYE + KLM,
        (668.36505710177, b'b', 7, 2),  	# LAKTYET + KLM,
        (718.88889610177, b'b', 8, 2),  	# LAKTYETT + KLM,
        (720.37740573677, b'y', 6, 1),  	# ETTLEK,
        (775.43092810177, b'b', 9, 2),  	# LAKTYETTL + KLM,
        (820.94439723677, b'y', 9, 2),  	# KTYETTLEK + KLM,
        (839.95222460177, b'b', 10, 2),  # LAKTYETTLE + KLM,
        (841.52155973677, b'b', 3, 1),  	# LAK + KLM,
        (856.46295423677, b'y', 10, 2),  # AKTYETTLEK + KLM,
        (883.44073473677, b'y', 7, 1),  	# YETTLEK,
        (942.56923773677, b'b', 4, 1),  	# LAKT + KLM,
        (984.48841273677, b'y', 8, 1),   # TYETTLEK,
        (1105.63256673677, b'b', 5, 1),  # LAKTY + KLM,
        (1234.67515973677, b'b', 6, 1),  # LAKTYE + KLM,
        (1335.72283773677, b'b', 7, 1),  # LAKTYET + KLM,
        (1436.77051573677, b'b', 8, 1),  # LAKTYETT + KLM,
        (1549.85457973677, b'b', 9, 1),  # LAKTYETTL + KLM,
        (1640.88151800677, b'y', 9, 1),  # KTYETTLEK + KLM,
        (1678.89717273677, b'b', 10, 1),  # LAKTYETTLE + KLM,
        (1711.91863200677, b'y', 10, 1),  # AKTYETTLEK + KLM,
    ], dtype=mock_frag_dtype)

    fragments = fragment_crosslinked_peptide(frag_pep_index=1,
                                             cl_pep_index=0,
                                             link_pos=2,
                                             crosslinker=ctx.config.crosslinker[0],
                                             context=ctx)
    fragments = spread_charges(fragments, ctx, 2)

    fragments.sort(order='mz')
    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])

    # add precursor
    precursor_fragments = np.array([
        # mz, ion_type, idx, charge
        (913.0049862, b'P', 11, 2),  # LAKTYETTLEK + KLM,
        (1825.002696, b'P', 11, 1),  # LAKTYETTLEK + KLM,
    ], dtype=mock_frag_dtype)

    expected_fragments = np.concatenate([expected_fragments, precursor_fragments])

    expected_fragments.sort(order='mz')

    fragments = fragment_crosslinked_peptide(frag_pep_index=1, cl_pep_index=0,
                                             link_pos=2,
                                             crosslinker=ctx.config.crosslinker[0], context=ctx,
                                             add_precursor=True)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()
    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])


def test_fragment_crosslinked_peptide_stubs():
    ctx = MockContext(Config(crosslinker=['DSSO']))

    # setup peptide_db
    peptides = np.array([b'KLM', b'NEIKALK'])
    ctx.setup_peptide_db(peptides)

    # fragments for NEIKALK (charge:2)
    # with KLM as second peptide linked on K4
    # generated Xi1 PeptideToIon.bat
    expected_fragments = np.array([
        # mz, charge, LN, loss, nlosses, stub, ion_type, term, idx, pep_id, ranges
        (58.02873996677, 2, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # b1
        (74.06003810177, 2, True, b'', 0, b'', b'y', b'c', 1, 1, ((6, 7), (0, 0))),  # y1
        (115.05020346677, 1, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # b1
        (122.55003646677, 2, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # b2
        (130.60207010177, 2, True, b'', 0, b'', b'y', b'c', 2, 1, ((5, 7), (0, 0))),  # y2
        (147.11279973677, 1, True, b'', 0, b'', b'y', b'c', 1, 1, ((6, 7), (0, 0))),  # y1
        (166.12062710177, 2, True, b'', 0, b'', b'y', b'c', 3, 1, ((4, 7), (0, 0))),  # y3
        (179.09206846677, 2, True, b'', 0, b'', b'b', b'n', 3, 1, ((0, 3), (0, 0))),  # b3
        (244.09279646677, 1, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # b2
        (257.17339095177, 2, True, b'', 0, b'a', b'y', b'c', 4, 1, ((3, 7), (0, 0))),  # y4_A
        (260.19686373677, 1, True, b'', 0, b'', b'y', b'c', 2, 1, ((5, 7), (0, 0))),  # y2
        (270.14483231677, 2, True, b'', 0, b'a', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # b4_A
        (273.15942630177, 2, True, b'', 0, b't', b'y', b'c', 4, 1, ((3, 7), (0, 0))),  # y4_T
        (282.16470865177, 2, True, b'', 0, b's', b'y', b'c', 4, 1, ((3, 7), (0, 0))),  # y4_S
        (286.13086766677, 2, True, b'', 0, b't', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # b4_T
        (295.13615001677, 2, True, b'', 0, b's', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # b4_S
        (305.66338931677, 2, True, b'', 0, b'a', b'b', b'n', 5, 1, ((0, 5), (0, 0))),  # b5_A
        (313.71542295177, 2, True, b'', 0, b'a', b'y', b'c', 5, 1, ((2, 7), (0, 0))),  # y5_A
        (321.64942466677, 2, True, b'', 0, b't', b'b', b'n', 5, 1, ((0, 5), (0, 0))),  # b5_T
        (329.70145830177, 2, True, b'', 0, b't', b'y', b'c', 5, 1, ((2, 7), (0, 0))),  # y5_T
        (330.65470701677, 2, True, b'', 0, b's', b'b', b'n', 5, 1, ((0, 5), (0, 0))),  # b5_S
        (331.23397773677, 1, True, b'', 0, b'', b'y', b'c', 3, 1, ((4, 7), (0, 0))),  # y3
        (338.70674065177, 2, True, b'', 0, b's', b'y', b'c', 5, 1, ((2, 7), (0, 0))),  # y5_S
        (357.17686046677, 1, True, b'', 0, b'', b'b', b'n', 3, 1, ((0, 3), (0, 0))),  # b3
        (362.20542131677, 2, True, b'', 0, b'a', b'b', b'n', 6, 1, ((0, 6), (0, 0))),  # b6_A
        (378.19145666677, 2, True, b'', 0, b't', b'b', b'n', 6, 1, ((0, 6), (0, 0))),  # b6_T
        (378.23671945177, 2, True, b'', 0, b'a', b'y', b'c', 6, 1, ((1, 7), (0, 0))),  # y6_A
        (387.19673901677, 2, True, b'', 0, b's', b'b', b'n', 6, 1, ((0, 6), (0, 0))),  # b6_S
        (394.22275480177, 2, True, b'', 0, b't', b'y', b'c', 6, 1, ((1, 7), (0, 0))),  # y6_T
        (403.22803715177, 2, True, b'', 0, b's', b'y', b'c', 6, 1, ((1, 7), (0, 0))),  # y6_S
        (435.25818295177, 2, True, b'', 0, b'a', b'P', b'P', 7, 1, ((0, 7), (0, 0))),  # P_A
        (451.24421830177, 2, True, b'', 0, b't', b'P', b'P', 7, 1, ((0, 7), (0, 0))),  # P_T
        (460.24950065177, 2, True, b'', 0, b's', b'P', b'P', 7, 1, ((0, 7), (0, 0))),  # P_S
        (504.28504473677, 2, False, b'', 0, b'', b'y', b'c', 4, 1, ((3, 7), (0, 3))),  # y4+P,
        (513.33950543677, 1, True, b'', 0, b'a', b'y', b'c', 4, 1, ((3, 7), (0, 0))),  # y4_A
        (517.25648610177, 2, False, b'', 0, b'', b'b', b'n', 4, 1, ((0, 4), (0, 3))),  # b4+P,
        (539.28238816677, 1, True, b'', 0, b'a', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # b4_A
        (545.31157613677, 1, True, b'', 0, b't', b'y', b'c', 4, 1, ((3, 7), (0, 0))),  # y4_T
        (552.77504310177, 2, False, b'', 0, b'', b'b', b'n', 5, 1, ((0, 5), (0, 3))),  # b5+P,
        (560.82707673677, 2, False, b'', 0, b'', b'y', b'c', 5, 1, ((2, 7), (0, 3))),  # y5+P,
        (563.32214083677, 1, True, b'', 0, b's', b'y', b'c', 4, 1, ((3, 7), (0, 0))),  # y4_S
        (571.25445886677, 1, True, b'', 0, b't', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # b4_T
        (589.26502356677, 1, True, b'', 0, b's', b'b', b'n', 4, 1, ((0, 4), (0, 0))),  # b4_S
        (609.31707510177, 2, False, b'', 0, b'', b'b', b'n', 6, 1, ((0, 6), (0, 3))),  # b6+P,
        (610.31950216677, 1, True, b'', 0, b'a', b'b', b'n', 5, 1, ((0, 5), (0, 0))),  # b5_A
        (625.34837323677, 2, False, b'', 0, b'', b'y', b'c', 6, 1, ((1, 7), (0, 3))),  # y6+P,
        (626.42356943677, 1, True, b'', 0, b'a', b'y', b'c', 5, 1, ((2, 7), (0, 0))),  # y5_A
        (642.29157286677, 1, True, b'', 0, b't', b'b', b'n', 5, 1, ((0, 5), (0, 0))),  # b5_T
        (658.39564013677, 1, True, b'', 0, b't', b'y', b'c', 5, 1, ((2, 7), (0, 0))),  # y5_T
        (660.30213756677, 1, True, b'', 0, b's', b'b', b'n', 5, 1, ((0, 5), (0, 0))),  # b5_S
        (676.40620483677, 1, True, b'', 0, b's', b'y', b'c', 5, 1, ((2, 7), (0, 0))),  # y5_S
        (682.36983673677, 2, False, b'', 0, b'', b'P', b'P', 7, 1, ((0, 7), (0, 3))),  # P+P,
        (723.40356616677, 1, True, b'', 0, b'a', b'b', b'n', 6, 1, ((0, 6), (0, 0))),  # b6_A
        (755.37563686677, 1, True, b'', 0, b't', b'b', b'n', 6, 1, ((0, 6), (0, 0))),  # b6_T
        (755.46616243677, 1, True, b'', 0, b'a', b'y', b'c', 6, 1, ((1, 7), (0, 0))),  # y6_A
        (773.38620156677, 1, True, b'', 0, b's', b'b', b'n', 6, 1, ((0, 6), (0, 0))),  # b6_S
        (787.43823313677, 1, True, b'', 0, b't', b'y', b'c', 6, 1, ((1, 7), (0, 0))),  # y6_T
        (805.44879783677, 1, True, b'', 0, b's', b'y', b'c', 6, 1, ((1, 7), (0, 0))),  # y6_S
        (869.50908943677, 1, True, b'', 0, b'a', b'P', b'P', 7, 1, ((0, 7), (0, 0))),  # P_A
        (901.48116013677, 1, True, b'', 0, b't', b'P', b'P', 7, 1, ((0, 7), (0, 0))),  # P_T
        (919.49172483677, 1, True, b'', 0, b's', b'P', b'P', 7, 1, ((0, 7), (0, 0))),  # P_S
        (1007.56281300677, 1, False, b'', 0, b'', b'y', b'c', 4, 1, ((3, 7), (0, 3))),  # y4+P,
        (1033.50569573677, 1, False, b'', 0, b'', b'b', b'n', 4, 1, ((0, 4), (0, 3))),  # b4+P,
        (1104.54280973677, 1, False, b'', 0, b'', b'b', b'n', 5, 1, ((0, 5), (0, 3))),  # b5+P,
        (1120.64687700677, 1, False, b'', 0, b'', b'y', b'c', 5, 1, ((2, 7), (0, 3))),  # y5+P,
        (1217.62687373677, 1, False, b'', 0, b'', b'b', b'n', 6, 1, ((0, 6), (0, 3))),  # b6+P,
        (1249.68947000677, 1, False, b'', 0, b'', b'y', b'c', 6, 1, ((1, 7), (0, 3))),  # y6+P,
        (1363.73239700677, 1, False, b'', 0, b'', b'P', b'P', 7, 1, ((0, 7), (0, 3))),  # P+P,
    ], dtype=dtypes.fragments)

    fragments = fragment_crosslinked_peptide(frag_pep_index=1, cl_pep_index=0,
                                             link_pos=3,
                                             crosslinker=ctx.config.crosslinker[0], context=ctx,
                                             add_precursor=True)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()
    for name in fragments.dtype.names:
        if np.issubdtype(fragments.dtype[name], np.number):
            np.testing.assert_allclose(
                fragments[name], expected_fragments[name], err_msg=f'{name} doesnae match, '
                f'expected_fragments {expected_fragments[name]}, got {fragments[name]}')
        else:
            assert np.all(fragments[name] == expected_fragments[name]), f'{name} doesnae match, ' \
                f'expected_fragments {expected_fragments[name]}, got {fragments[name]}'


def test_fragment_crosslinked_peptide_minimal_stubs():
    ctx = MockContext(Config(crosslinker=['DSSO']))

    # setup peptide_db
    peptides = np.array([b'KLM', b'NEIKALK'])
    ctx.setup_peptide_db(peptides)

    # fragments for NEIKALK (charge:2)
    # with KLM as second peptide linked on K4
    # generated Xi1 PeptideToIon.bat
    expected_fragments = np.array([
        # mz, LN, stub
        (115.05020346677, True, 0),  # b1
        (147.11279973677, True, 0),  # y1
        (244.09279646677, True, 0),  # b2
        (260.19686373677, True, 0),  # y2
        (331.23397773677, True, 0),  # y3
        (357.17686046677, True, 0),  # b3
        (513.33950543677, True, 54.010565),  # y4_A
        (539.28238816677, True, 54.010565),  # b4_A
        (545.31157613677, True, 85.982635),  # y4_T
        (563.32214083677, True, 103.9932),  # y4_S
        (571.25445886677, True, 85.982635),  # b4_T
        (589.26502356677, True, 103.9932),  # b4_S
        (610.31950216677, True, 54.010565),  # b5_A
        (626.42356943677, True, 54.010565),  # y5_A
        (642.29157286677, True, 85.982635),  # b5_T
        (658.39564013677, True, 85.982635),  # y5_T
        (660.30213756677, True, 103.9932),  # b5_S
        (676.40620483677, True, 103.9932),  # y5_S
        (723.40356616677, True, 54.010565),  # b6_A
        (755.37563686677, True, 85.982635),  # b6_T
        (755.46616243677, True, 54.010565),  # y6_A
        (773.38620156677, True, 103.9932),  # b6_S
        (787.43823313677, True, 85.982635),  # y6_T
        (805.44879783677, True, 103.9932),  # y6_S
        (869.50908943677, True, 54.010565),  # P_A
        (901.48116013677, True, 85.982635),  # P_T
        (919.49172483677, True, 103.9932),  # P_S
        (1007.56281300677, False, 0),  # y4+P
        (1033.50569573677, False, 0),  # b4+P
        (1104.54280973677, False, 0),  # b5+P
        (1120.64687700677, False, 0),  # y5+P
        (1217.62687373677, False, 0),  # b6+P
        (1249.68947000677, False, 0)  # y6+P
    ], dtype=[('mz', np.float64), ('LN', bool), ('stub', np.float64)])

    mz, linearity, stub = fragment_crosslinked_peptide_minimal(
        frag_pep_index=1, cl_pep_index=0, link_pos=[3], crosslinker=ctx.config.crosslinker[0],
        context=ctx)

    order = np.argsort(mz)
    mz = mz[order]
    linearity = linearity[order]
    stub = stub[order]

    np.testing.assert_allclose(expected_fragments['mz'], mz)
    np.testing.assert_equal(expected_fragments['LN'], linearity)
    np.testing.assert_allclose(expected_fragments['stub'], stub)


def test_fragment_modified_crosslinked_peptide():

    test_mod1 = Modification(name='cm', specificity=['C'], type='fixed', composition="C2H3O1N1")
    test_mod2 = Modification(name='ox', specificity=['M'], type='variable', composition="O1")
    ctx = MockContext(
        Config(crosslinker=['BS3'],
               modification=ModificationConfig(modifications=[test_mod1, test_mod2]))
    )

    # setup peptide_db
    peptides = np.array([b'KLM', b'oxMAcmCK'])
    ctx.setup_peptide_db(peptides)

    # fragments for MACK (charge:2)
    # with carbamidomethylation modification on C and oxidation on M
    # with KLM as second peptide linked on K3
    # generated Xi1 PeptideToIon.bat
    expected_fragments = np.array([
        # mz, ion_type, idx, charge
        (74.52497396677, b'b', 1, 2),    # MoxA
        (110.04353096677, b'b', 2, 2),   # MoxA
        (148.04267146677, b'b', 1, 1),   # MoxA
        (190.05885596677, b'b', 3, 2),   # MoxACcm
        (219.07978546677, b'b', 2, 1),   # MoxA
        (338.20910923677, b'y', 1, 2),   # K + KLM
        (379.11043546677, b'b', 3, 1),   # MoxACcm
        (418.22443423677, b'y', 2, 2),   # CcmK + KLM
        (453.74299123677, b'y', 3, 2),   # ACcmK + KLM
        (675.41094200677, b'y', 1, 1),   # K + KLM
        (835.44159200677, b'y', 2, 1),   # CcmK + KLM
        (906.47870600677, b'y', 3, 1),   # ACcmK + KLM
    ], dtype=mock_frag_dtype)

    fragments = fragment_crosslinked_peptide(frag_pep_index=1, cl_pep_index=0,
                                             link_pos=3,
                                             crosslinker=ctx.config.crosslinker[0], context=ctx)
    fragments = spread_charges(fragments, ctx, 2)

    fragments.sort()
    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])


def test_fragment_cn_terminal_modified_crosslinked_peptide():

    n_term_mod = Modification(name='acetyl-', specificity=['X'], type='fixed', composition="C2H2O1")
    c_term_mod = Modification(name='-amide', specificity=['X'], type='fixed', composition="N1H1O-1")

    ctx = MockContext(
        Config(crosslinker=['BS3'],
               modification=ModificationConfig(modifications=[n_term_mod, c_term_mod])))

    # setup peptide_db
    peptides = np.array([b'acetyl-AKA-amide', b'KLM'])
    ctx.setup_peptide_db(peptides)

    # fragments for AKA (charge:2)
    # with n-terminal acetyl and c-terminal amide
    # with KLM as second peptide linked on K1
    # generated Xi1 PeptideToIon.bat
    expected_fragments = np.array([
        # mz, ion_type, idx, charge
        (57.53111596677, b'b', 1, 2),        # acetyl-A
        (114.05495546677, b'b', 1, 1),       # acetyl-A
        (385.72766860177, b'b', 2, 2),       # acetyl-AK + KLM
        (770.44806073677, b'b', 2, 1),       # acetyl-AK + KLM
        (45.039105601770004, b'y', 1, 2),    # A-amide
        (89.07093473677, b'y', 1, 1),        # A-amide
        (373.23565823677006, b'y', 2, 2),    # KA-amide + KLM
        (745.4640400067701, b'y', 2, 1),     # KA-amide + KLM
    ], dtype=mock_frag_dtype)

    expected_fragments.sort(order='mz')

    fragments = fragment_crosslinked_peptide(frag_pep_index=0, cl_pep_index=1,
                                             link_pos=1,
                                             crosslinker=ctx.config.crosslinker[0], context=ctx)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()
    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])

    # add precursor
    precursor_fragments = np.array([
        # mz, ion_type, idx, charge
        (429.75949773677, b'P', 3, 2),  # acetyl-AKA-amide + KLM,
        (858.51171900677, b'P', 3, 1),  # acetyl-AKA-amide + KLM,
    ], dtype=mock_frag_dtype)

    expected_fragments = np.concatenate([expected_fragments, precursor_fragments])

    expected_fragments.sort(order='mz')

    fragments = fragment_crosslinked_peptide(frag_pep_index=0, cl_pep_index=1,
                                             link_pos=1,
                                             crosslinker=ctx.config.crosslinker[0], context=ctx,
                                             add_precursor=True)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()
    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])


def test_fragment_noncovalent_peptide_pair():
    """Test fragmenting a noncovalently bound peptide pair with losses"""
    config = Config(crosslinker=Crosslinker.BS3, ms2_tol='1 ppm',
                    fragmentation=FragmentationConfig(losses=[Loss.H2O], max_nloss=1))

    ctx = MockContext(config)

    # setup peptide_db
    peptides = np.array([b'HQME', b'STAY'])
    ctx.setup_peptide_db(peptides)

    # fragments for non covalent STAY HOME (charge:2)
    # generated with PeptidesToIon.sh (xi1)
    expected_fragments = np.array([
        # (mz, ion_type, idx, charge)  # seq
        (88.03930446677, b'b', 1, 1),  # S
        (44.52329046677, b'b', 1, 2),  # S
        (189.08698246677, b'b', 2, 1),  # ST
        (95.04712946677, b'b', 2, 2),  # ST
        (260.12409646677, b'b', 3, 1),  # STA
        (130.56568646677, b'b', 3, 2),  # STA
        (354.16595773677, b'y', 3, 1),  # TAY
        (177.58661710177, b'y', 3, 2),  # TAY
        (253.11827973677, b'y', 2, 1),  # AY
        (127.06277810177, b'y', 2, 2),  # AY
        (182.08116573677, b'y', 1, 1),  # Y
        (91.54422110177, b'y', 1, 2),  # Y
        (441.19798573677, b'P', 4, 1),  # STAY
        (221.10263110177, b'P', 4, 2),  # STAY
        (70.02874419677, b'b', 1, 1),  # S_H20x1
        (35.51801033177, b'b', 1, 2),  # S_H20x1
        (171.07642219677, b'b', 2, 1),  # ST_H20x1
        (86.04184933177, b'b', 2, 2),  # ST_H20x1
        (242.11353619677, b'b', 3, 1),  # STA_H20x1
        (121.56040633177, b'b', 3, 2),  # STA_H20x1
        (336.15539746677, b'y', 3, 1),  # TAY_H20x1
        (168.58133696677, b'y', 3, 2),  # TAY_H20x1
        (235.10771946677, b'y', 2, 1),  # AY_H20x1
        (118.05749796677, b'y', 2, 2),  # AY_H20x1
        (164.07060546677, b'y', 1, 1),  # Y_H20x1
        (82.53894096677, b'y', 1, 2),  # Y_H20x1
        (423.18742546677, b'P', 4, 1),  # STAY_H20x1
        (212.09735096677, b'P', 4, 2),  # STAY_H20x1
        (138.06618846677, b'b', 1, 1),  # H
        (69.53673246677, b'b', 1, 2),  # H
        (266.12476646677, b'b', 2, 1),  # HQ
        (133.56602146677, b'b', 2, 2),  # HQ
        (397.16525146677, b'b', 3, 1),  # HQM
        (199.08626396677, b'b', 3, 2),  # HQM
        (407.15949273677, b'y', 3, 1),  # QME
        (204.08338460177, b'y', 3, 2),  # QME
        (279.10091473677, b'y', 2, 1),  # ME
        (140.05409560177, b'y', 2, 2),  # ME
        (148.06042973677, b'y', 1, 1),  # E
        (74.53385310177, b'y', 1, 2),  # E
        (544.21840473677, b'P', 4, 1),  # HQME
        (272.61284060177, b'P', 4, 2),  # HQME
        (389.14893246677, b'y', 3, 1),  # QME_H20x1
        (195.07810446677, b'y', 3, 2),  # QME_H20x1
        (261.09035446677, b'y', 2, 1),  # ME_H20x1
        (131.04881546677, b'y', 2, 2),  # ME_H20x1
        (130.04986946677, b'y', 1, 1),  # E_H20x1
        (65.52857296677, b'y', 1, 2),  # E_H20x1
        (526.20784446677, b'P', 4, 1),  # HQME_H20x1
        (263.60756046677, b'P', 4, 2),  # HQME_H20x1
    ], dtype=mock_frag_dtype)

    expected_fragments.sort(order='mz')

    fragments = fragment_noncovalent_peptide_pair(pep1_index=1, pep2_index=0,
                                                  context=ctx, add_precursor=True)
    fragments = include_losses(fragments, [1, 0], ctx)
    fragments = spread_charges(fragments, ctx, 2)
    fragments.sort()

    np.testing.assert_almost_equal(fragments['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(fragments['ion_type'], expected_fragments['ion_type'])
    np.testing.assert_equal(fragments['idx'], expected_fragments['idx'])
    np.testing.assert_equal(fragments['charge'], expected_fragments['charge'])


def test_cython_fragmentation(tmpdir):
    var_mod = Modification(name='ox', specificity=['M'], type='variable', mass=15.99491463)
    config = Config(modification=ModificationConfig(modifications=[var_mod]))
    config.fragmentation.add_precursor = True
    ctx = MockContext(config)
    peptides = np.array([b'KoxMCC'])
    ctx.setup_peptide_db(peptides)

    fragmentation = Fragmentation(ctx)
    fragments = fragmentation.fragment(b'KMCC', np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8), 4, 1,
                                       ctx.peptide_db.peptide_mass(0) + const.PROTON_MASS)

    fragments.sort(order=['ion_type', 'idx'])
    expected_fragments = np.array([
        # mz           ,charge,LN,loss,nlosses,stub,ion_type,term,idx,pep_id,ranges
        (129.10223946677, 1, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # b1
        (276.13763446677, 1, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # b2
        (379.14681846677, 1, True, b'', 0, b'', b'b', b'n', 3, 1, ((0, 3), (0, 0))),  # b3
        (372.07159973677, 1, True, b'', 0, b'', b'y', b'c', 3, 1, ((1, 4), (0, 0))),  # y3
        (225.03620473677, 1, True, b'', 0, b'', b'y', b'c', 2, 1, ((2, 4), (0, 0))),  # y2
        (122.02702073677, 1, True, b'', 0, b'', b'y', b'c', 1, 1, ((3, 4), (0, 0))),  # y1
        (500.16656273677, 1, True, b'', 0, b'', b'P', b'P', 4, 1, ((0, 4), (0, 0)))   # peptide
    ], dtype=dtypes.fragments)
    expected_fragments.sort(order=['ion_type', 'idx'])

    for name in fragments.dtype.names:
        if np.issubdtype(fragments.dtype[name], np.number):
            np.testing.assert_allclose(
                fragments[name], expected_fragments[name], rtol=2e-7,
                err_msg=f'{name} doesnae match, '
                        f'expected_fragments {expected_fragments[name]}, got {fragments[name]}')
        else:
            assert np.all(fragments[name] == expected_fragments[name]), \
                f'{name} doesnae match, ' \
                f'expected_fragments {expected_fragments[name]}, got {fragments[name]}'


def test_cython_fragmentation_term_mods(tmpdir):
    var_mod = [Modification(name='ox', specificity=['M'], type='variable', mass=15.99491463),
               Modification(name='ac-', specificity=['X'], type='variable', composition='H2C2O1'),
               Modification(name='-1', specificity=['X'], type='variable', mass=1)]
    config = Config(modification=ModificationConfig(modifications=var_mod),
                    fragmentation=FragmentationConfig(nterm_ions=['a', 'b', 'c'],
                                                      cterm_ions=['x', 'y', 'z']))
    config.fragmentation.add_precursor = True
    ctx = MockContext(config)
    peptides = np.array([b'ac-KoxMCC-1'])
    ctx.setup_peptide_db(peptides)

    fragmentation = Fragmentation(ctx)
    fragments = fragmentation.fragment(b'KMCC', np.array([2, 3, 0, 1, 0, 0], dtype=np.uint8), 4, 1,
                                       ctx.peptide_db.peptide_mass(0) + const.PROTON_MASS)

    fragments.sort(order=['ion_type', 'idx'])
    expected_fragments = np.array([
        # mz           ,charge,LN,loss,nlosses,stub,ion_type,term,idx,pep_id,ranges
        (143.11789446677, 1, True, b'', 0, b'', b'a', b'n', 1, 1, ((0, 1), (0, 0))),  # ac-K, a1
        (290.15328946677, 1, True, b'', 0, b'', b'a', b'n', 2, 1, ((0, 2), (0, 0))),  # ac-KoxM, a2
        (393.16247346677, 1, True, b'', 0, b'', b'a', b'n', 3, 1, ((0, 3), (0, 0))),  # ac-KoxMC, a3
        (171.11280446677, 1, True, b'', 0, b'', b'b', b'n', 1, 1, ((0, 1), (0, 0))),  # ac-K, b1
        (318.14819946677, 1, True, b'', 0, b'', b'b', b'n', 2, 1, ((0, 2), (0, 0))),  # ac-KoxM, b2
        (421.15738346677, 1, True, b'', 0, b'', b'b', b'n', 3, 1, ((0, 3), (0, 0))),  # ac-KoxMC, b3
        (188.13935357177002, 1, True, b'', 0, b'', b'c', b'n', 1, 1, ((0, 1), (0, 0))),  # ac-K, c1
        (335.17474857177, 1, True, b'', 0, b'', b'c', b'n', 2, 1, ((0, 2), (0, 0))),  # ac-KoxM, c2
        (438.18393257177, 1, True, b'', 0, b'', b'c', b'n', 3, 1, ((0, 3), (0, 0))),  # ac-KoxMC, c3
        (399.05085946677, 1, True, b'', 0, b'', b'x', b'c', 3, 1, ((1, 4), (0, 0))),  # oxMCC-1, x3
        (252.01546446677003, 1, True, b'', 0, b'', b'x', b'c', 2, 1, ((2, 4), (0, 0))),  # CC-1, x2
        (149.00628046677002, 1, True, b'', 0, b'', b'x', b'c', 1, 1, ((3, 4), (0, 0))),  # C-1, x1
        (373.07159973677, 1, True, b'', 0, b'', b'y', b'c', 3, 1, ((1, 4), (0, 0))),  # oxMCC-1, y3
        (226.03620473677, 1, True, b'', 0, b'', b'y', b'c', 2, 1, ((2, 4), (0, 0))),  # CC-1, y2
        (123.02702073677, 1, True, b'', 0, b'', b'y', b'c', 1, 1, ((3, 4), (0, 0))),  # C-1, y1
        (356.04505043177, 1, True, b'', 0, b'', b'z', b'c', 3, 1, ((1, 4), (0, 0))),  # oxMCC-1, z3
        (209.00965543177003, 1, True, b'', 0, b'', b'z', b'c', 2, 1, ((2, 4), (0, 0))),  # CC-1, z2
        (106.00047143177, 1, True, b'', 0, b'', b'z', b'c', 1, 1, ((3, 4), (0, 0))),  # C-1, z1
        (543.17712773677, 1, True, b'', 0, b'', b'P', b'P', 4, 1, ((0, 4), (0, 0))),  # peptide
    ], dtype=dtypes.fragments)
    expected_fragments.sort(order=['ion_type', 'idx'])

    compare_numpy(expected_fragments, fragments, rtol=2e-7)
