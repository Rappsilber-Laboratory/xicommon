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

from xicommon.fragmentation import fragment_sequences, fragment_ions, spread_charges, \
    include_losses
from xicommon.config import Config, ModificationConfig, Modification, Loss, Crosslinker, \
    FragmentationConfig
from xicommon.mock_context import MockContext
from xicommon import const
import numpy as np
from numpy.testing import assert_array_equal


def test_fragment_ions_no_losses():
    # setup config and context
    mod = Modification(name='m', specificity=['C'], type='fixed', composition="H1")
    config = Config(modification=ModificationConfig(modifications=[mod]))
    ctx = MockContext(config)

    unmodified_sequences = np.array([b'LACK', b'LAG'])

    # modified_sequences = np.array([b'LAmCK', b'LAG'])
    mod_peptides = np.array([
        (0, [0, 0, 0, 0, 1, 0]),
        (1, [0, 0, 0, 0, 0, 0]),
    ], dtype=[('sequence_index', '<i8'), ('modifications', 'u1', (6,))])

    frag_generator = fragment_ions(unmodified_sequences, mod_peptides, ctx)

    # generator yields once for each ion type, loss type and loss count
    fragments = list(frag_generator)
    nterm_frags = fragments[0]

    sites_dtype = [('peptide_index', '<i8'), ('fragment_index', '<i8'), ('nterm', '?'),
                   ('cterm', '?'), ('start', '<i8'), ('end', '<i8')]
    expected_nterm_sites = np.array([
        (0, 5, True, False, 0, 1),  # L - b1 of LACK
        (0, 6, True, False, 0, 2),  # LA - b2 of LACK
        (0, 7, True, False, 0, 3),  # LAC - b3 of LACK
        (1, 5, True, False, 0, 1),  # L - b1 of LAG
        (1, 6, True, False, 0, 2),  # LA - b2 of LAG
    ], dtype=sites_dtype)

    # yields term, ion, loss, loss_count, sites, masses
    assert nterm_frags[0] == b'n'    # term
    assert nterm_frags[1] == b'b'    # ion type
    assert nterm_frags[2] is None   # loss
    assert nterm_frags[3] == 0      # loss count
    assert_array_equal(nterm_frags[4], expected_nterm_sites)       # sites


def test_fragment_sequences_single():
    peptide_sequences = np.array([b'LVTDLTK'])

    expected_fragments = np.array([(0, 1, b'L'),
                                   (0, 2, b'LV'),
                                   (0, 3, b'LVT'),
                                   (0, 4, b'LVTD'),
                                   (0, 5, b'LVTDL'),
                                   (0, 6, b'LVTDLT'),
                                   (1, 7, b'VTDLTK'),
                                   (2, 7, b'TDLTK'),
                                   (3, 7, b'DLTK'),
                                   (4, 7, b'LTK'),
                                   (5, 7, b'TK'),
                                   (6, 7, b'K'),
                                   ], dtype=[('start', np.intp),
                                             ('end', np.intp),
                                             ('sequence', '|S7')])

    fragments, sites = fragment_sequences(peptide_sequences, add_precursor=False)

    sites.sort(order=['start', 'end'])
    idx = sites['fragment_index']

    assert np.all(sites['peptide_index'] == 0)
    assert_array_equal(expected_fragments['start'], sites['start'])
    assert_array_equal(expected_fragments['end'], sites['end'])
    assert_array_equal(expected_fragments['sequence'], fragments[idx])


def test_fragment_sequences_single_with_precursor():
    peptide_sequences = np.array([b'LVTDLTK'])

    expected_fragments = np.array([(0, 1, b'L'),
                                   (0, 2, b'LV'),
                                   (0, 3, b'LVT'),
                                   (0, 4, b'LVTD'),
                                   (0, 5, b'LVTDL'),
                                   (0, 6, b'LVTDLT'),
                                   (0, 7, b'LVTDLTK'),  # this is the precursor
                                   (1, 7, b'VTDLTK'),
                                   (2, 7, b'TDLTK'),
                                   (3, 7, b'DLTK'),
                                   (4, 7, b'LTK'),
                                   (5, 7, b'TK'),
                                   (6, 7, b'K'),
                                   ], dtype=[('start', np.intp),
                                             ('end', np.intp),
                                             ('sequence', '|S7')])
    prec_row = 6
    fragments, sites = fragment_sequences(peptide_sequences, add_precursor=True)

    sites.sort(order=['start', 'end'])
    idx = sites['fragment_index']

    assert np.all(sites['peptide_index'] == 0)
    assert_array_equal(expected_fragments['start'], sites['start'])
    assert_array_equal(expected_fragments['end'], sites['end'])
    assert_array_equal(expected_fragments['sequence'], fragments[idx])

    # precursor tests
    assert sites['nterm'][prec_row] == 0
    assert sites['cterm'][prec_row] == 0
    assert sites['start'][prec_row] == 0
    assert sites['end'][prec_row] == len(peptide_sequences[0])


def test_fragment_sequences_multiple():

    peptide_sequences = np.array([b'LVTDLTK', b'LVTLTK'])

    expected_fragments = np.array([(0, 0, 1, b'L'),
                                   (0, 0, 2, b'LV'),
                                   (0, 0, 3, b'LVT'),
                                   (0, 0, 4, b'LVTD'),
                                   (0, 0, 5, b'LVTDL'),
                                   (0, 0, 6, b'LVTDLT'),
                                   (0, 1, 7, b'VTDLTK'),
                                   (0, 2, 7, b'TDLTK'),
                                   (0, 3, 7, b'DLTK'),
                                   (0, 4, 7, b'LTK'),
                                   (0, 5, 7, b'TK'),
                                   (0, 6, 7, b'K'),
                                   (1, 0, 1, b'L'),
                                   (1, 0, 2, b'LV'),
                                   (1, 0, 3, b'LVT'),
                                   (1, 0, 4, b'LVTL'),
                                   (1, 0, 5, b'LVTLT'),
                                   (1, 1, 6, b'VTLTK'),
                                   (1, 2, 6, b'TLTK'),
                                   (1, 3, 6, b'LTK'),
                                   (1, 4, 6, b'TK'),
                                   (1, 5, 6, b'K'),
                                   ], dtype=[('peptide_index', np.intp),
                                             ('start', np.intp),
                                             ('end', np.intp),
                                             ('sequence', '|S7')])

    fragments, sites = fragment_sequences(peptide_sequences, add_precursor=False)

    sites.sort(order=['peptide_index', 'start', 'end'])
    idx = sites['fragment_index']

    assert_array_equal(expected_fragments['peptide_index'], sites['peptide_index'])
    assert_array_equal(expected_fragments['start'], sites['start'])
    assert_array_equal(expected_fragments['end'], sites['end'])
    assert_array_equal(expected_fragments['sequence'], fragments[idx])
    assert len(fragments) == len(np.unique(expected_fragments['sequence']))


def test_fragment_sequences_multiple_with_precursor():

    peptide_sequences = np.array([b'LVTDLTK', b'LVTLTK'])

    expected_fragments = np.array([(0, 0, 1, b'L'),
                                   (0, 0, 2, b'LV'),
                                   (0, 0, 3, b'LVT'),
                                   (0, 0, 4, b'LVTD'),
                                   (0, 0, 5, b'LVTDL'),
                                   (0, 0, 6, b'LVTDLT'),
                                   (0, 0, 7, b'LVTDLTK'),  # this is the precursor
                                   (0, 1, 7, b'VTDLTK'),
                                   (0, 2, 7, b'TDLTK'),
                                   (0, 3, 7, b'DLTK'),
                                   (0, 4, 7, b'LTK'),
                                   (0, 5, 7, b'TK'),
                                   (0, 6, 7, b'K'),
                                   (1, 0, 1, b'L'),
                                   (1, 0, 2, b'LV'),
                                   (1, 0, 3, b'LVT'),
                                   (1, 0, 4, b'LVTL'),
                                   (1, 0, 5, b'LVTLT'),
                                   (1, 0, 6, b'LVTLTK'),  # this is the precursor
                                   (1, 1, 6, b'VTLTK'),
                                   (1, 2, 6, b'TLTK'),
                                   (1, 3, 6, b'LTK'),
                                   (1, 4, 6, b'TK'),
                                   (1, 5, 6, b'K'),
                                   ], dtype=[('peptide_index', np.intp),
                                             ('start', np.intp),
                                             ('end', np.intp),
                                             ('sequence', '|S7')])
    # positions of the precursors in the table
    prec_row = [6, 18]
    fragments, sites = fragment_sequences(peptide_sequences, add_precursor=True)

    sites.sort(order=['peptide_index', 'start', 'end'])
    idx = sites['fragment_index']

    assert_array_equal(expected_fragments['peptide_index'], sites['peptide_index'])
    assert_array_equal(expected_fragments['start'], sites['start'])
    assert_array_equal(expected_fragments['end'], sites['end'])
    assert_array_equal(expected_fragments['sequence'], fragments[idx])
    assert len(fragments) == len(np.unique(expected_fragments['sequence']))

    # precursor tests
    assert np.all(sites['nterm'][prec_row] == [False, False])
    assert np.all(sites['cterm'][prec_row] == [False, False])
    assert np.all(sites['start'][prec_row] == [0, 0])
    assert np.all(sites['end'][prec_row] == np.char.str_len(peptide_sequences))


def test_spread_charges_nofilter():
    # test with two linear and two cross-linked fragments
    input = np.array([(1, True, 1), (2, True, 1), (3, False, 1), (4, False, 1),
                      ], dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected = np.array([
        # singly charged
        (1, True, 1), (2, True, 1), (3, False, 1), (4, False, 1),
        # doubly charged
        ((1 + const.PROTON_MASS)/2, True, 2), ((2 + const.PROTON_MASS)/2, True, 2),
        ((3 + const.PROTON_MASS)/2, False, 2), ((4 + const.PROTON_MASS)/2, False, 2),
        # triply charged
        ((1 + 2 * const.PROTON_MASS)/3, True, 3), ((2 + 2 * const.PROTON_MASS)/3, True, 3),
        ((3 + 2 * const.PROTON_MASS) / 3, False, 3), ((4 + 2 * const.PROTON_MASS) / 3, False, 3)],
        dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected.sort()

    output = spread_charges(input, "", 3)

    np.testing.assert_almost_equal(expected['mz'], output['mz'])
    np.testing.assert_equal(expected[['LN', 'charge']], output[['LN', 'charge']])


def test_spread_charges_filter():
    # test with two linear and two cross-linked fragments
    input = np.array([(1, True, 1), (2, True, 1), (3, False, 1), (4, False, 1),
                      ], dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected = np.array([
        # singly charged linear only
        (1, True, 1), (2, True, 1),
        # doubly charged linear and cross-linked
        ((1 + const.PROTON_MASS)/2, True, 2), ((2 + const.PROTON_MASS)/2, True, 2),
        ((3 + const.PROTON_MASS)/2, False, 2), ((4 + const.PROTON_MASS)/2, False, 2),
        # triply charged only cross-linked
        ((3 + 2 * const.PROTON_MASS) / 3, False, 3), ((4 + 2 * const.PROTON_MASS) / 3, False, 3)],
        dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected.sort()

    output = spread_charges(input, "", 3, max_linear_charge=2, min_crosslinked_charge=2)
    output.sort()
    np.testing.assert_almost_equal(expected['mz'], output['mz'])
    np.testing.assert_equal(expected[['LN', 'charge']], output[['LN', 'charge']])


def test_spread_charges_filter2():
    """Test spread_charges with only min_crosslinked_charge argument."""
    # test with two linear and two cross-linked fragments
    input = np.array([(1, True, 1), (2, True, 1), (3, False, 1), (4, False, 1),
                      ], dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected = np.array([
        # singly charged linear only
        (1, True, 1), (2, True, 1),
        # doubly charged
        ((1 + const.PROTON_MASS)/2, True, 2), ((2 + const.PROTON_MASS)/2, True, 2),
        ((3 + const.PROTON_MASS)/2, False, 2), ((4 + const.PROTON_MASS)/2, False, 2),
        # triply charged
        ((1 + 2 * const.PROTON_MASS)/3, True, 3), ((2 + 2 * const.PROTON_MASS)/3, True, 3),
        ((3 + 2 * const.PROTON_MASS) / 3, False, 3), ((4 + 2 * const.PROTON_MASS) / 3, False, 3)],
        dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected.sort()

    output = spread_charges(input, "", 3, min_crosslinked_charge=2)
    output.sort()
    np.testing.assert_almost_equal(expected['mz'], output['mz'])
    np.testing.assert_equal(expected[['LN', 'charge']], output[['LN', 'charge']])


def test_spread_charges_filter3():
    """Test spread_charges with only max_linear_charge argument."""
    # test with two linear and two cross-linked fragments
    input = np.array([(1, True, 1), (2, True, 1), (3, False, 1), (4, False, 1),
                      ], dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected = np.array([
        # singly charged
        (1, True, 1), (2, True, 1), (3, False, 1), (4, False, 1),
        # doubly charged
        ((1 + const.PROTON_MASS)/2, True, 2), ((2 + const.PROTON_MASS)/2, True, 2),
        ((3 + const.PROTON_MASS)/2, False, 2), ((4 + const.PROTON_MASS)/2, False, 2),
        # triply charged crosslinked only
        ((3 + 2 * const.PROTON_MASS) / 3, False, 3), ((4 + 2 * const.PROTON_MASS) / 3, False, 3)],
        dtype=[('mz', np.float64), ('LN', bool), ('charge', np.int8)])

    expected.sort()

    output = spread_charges(input, "", 3, max_linear_charge=2)
    output.sort()
    np.testing.assert_almost_equal(expected['mz'], output['mz'])
    np.testing.assert_equal(expected[['LN', 'charge']], output[['LN', 'charge']])


def test_include_losses():
    # columns needed: term, LN, nlosses, ranges, loss, mz
    # as include losses is kind of a thin wrapper around create_loss_fragments this is
    # functioning as a test for create_loss_fragments  as well

    water_loss = Loss(name='H2O', composition="H2O1", specificity=['L'])

    mock_frag_dtype = np.dtype([
        ('mz', np.float64), ('ion_type', 'S1'), ('idx', np.uint8),
        ('charge', np.uint8),
        ('nlosses', np.uint8),
        ('loss', 'S25'),
        ('ranges', np.uint8, (2, 2))
    ])

    ctx = MockContext(
        Config(crosslinker=Crosslinker.BS3, ms2_tol='1ppm',
               fragmentation=FragmentationConfig(losses=[water_loss], max_nloss=2))
    )
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))

    in_fragments = np.array([
        (129.10223948, b'b', 1, 1, 0, b'', ((0, 1), (0, 0))),  # K
        (182.08116968, b'y', 1, 1, 0, b'', ((0, 0), (3, 4))),  # Y
        (242.18630346, b'b', 2, 1, 0, b'', ((0, 2), (0, 0))),  # KL
        (310.1761327, b'y', 2, 1, 0, b'', ((0, 0), (2, 4))),  # KY
        (381.21324648, b'y', 3, 1, 0, b'', ((0, 0), (1, 4))),  # AKY
        (642.38949664, b'b', 1, 1, 0, b'', ((0, 3), (0, 1))),  # L - KLM
        (713.42661043, b'b', 2, 1, 0, b'', ((0, 3), (0, 2))),  # LA - KLM
        (781.41643967, b'y', 1, 1, 0, b'', ((2, 3), (0, 4))),  # M - LAKY
        (841.52157344, b'b', 3, 1, 0, b'', ((0, 3), (0, 3))),  # LAK - KLM
        (894.50050364, b'y', 2, 1, 0, b'', ((1, 3), (0, 4))),  # LM - LAKY
        # mock loss
        (900, b'y', 2, 1, 1, b'MCK', ((1, 3), (0, 4))),  # LM - LAKY
    ], dtype=mock_frag_dtype)

    expected_fragments = np.array([
        (129.10223948, b'b', 1, 1, 0, b'', ((0, 1), (0, 0))),  # K
        (182.08116968, b'y', 1, 1, 0, b'', ((0, 0), (3, 4))),  # Y
        (242.18630346, b'b', 2, 1, 0, b'', ((0, 2), (0, 0))),  # KL
        (310.1761327, b'y', 2, 1, 0, b'', ((0, 0), (2, 4))),  # KY
        (381.21324648, b'y', 3, 1, 0, b'', ((0, 0), (1, 4))),  # AKY
        (642.38949664, b'b', 1, 1, 0, b'', ((0, 3), (0, 1))),  # L - KLM
        (713.42661043, b'b', 2, 1, 0, b'', ((0, 3), (0, 2))),  # LA - KLM
        (781.41643967, b'y', 1, 1, 0, b'', ((2, 3), (0, 4))),  # M - LAKY
        (841.52157344, b'b', 3, 1, 0, b'', ((0, 3), (0, 3))),  # LAK - KLM
        (894.50050364, b'y', 2, 1, 0, b'', ((1, 3), (0, 4))),  # LM - LAKY

        (242.18630346 - 18.01056027, b'b', 2, 1, 1, b'H2Ox1', ((0, 2), (0, 0))),  # KL -H2O
        (642.38949664 - 18.01056027, b'b', 1, 1, 1, b'H2Ox1', ((0, 3), (0, 1))),  # L - KLM-H2O
        (713.42661043 - 18.01056027, b'b', 2, 1, 1, b'H2Ox1', ((0, 3), (0, 2))),  # LA - KLM-H2O
        (781.41643967 - 18.01056027, b'y', 1, 1, 1, b'H2Ox1', ((2, 3), (0, 4))),  # M - LAKY-H2O
        (841.52157344 - 18.01056027, b'b', 3, 1, 1, b'H2Ox1', ((0, 3), (0, 3))),  # LAK - KLM-H2O
        (894.50050364 - 18.01056027, b'y', 2, 1, 1, b'H2Ox1', ((1, 3), (0, 4))),  # LM - LAKY-H2O
        (642.38949664 - 36.02112054, b'b', 1, 1, 2, b'H2Ox2', ((0, 3), (0, 1))),  # L - KLM-H2Ox2
        (713.42661043 - 36.02112054, b'b', 2, 1, 2, b'H2Ox2', ((0, 3), (0, 2))),  # LA - KLM-H2Ox2
        (841.52157344 - 36.02112054, b'b', 3, 1, 2, b'H2Ox2', ((0, 3), (0, 3))),  # LAK - KLM-H2Ox2
        (894.50050364 - 36.02112054, b'y', 2, 1, 2, b'H2Ox2', ((1, 3), (0, 4))),  # LM - LAKY-H2Ox2

        # mock losses
        (900, b'y', 2, 1, 1, b'MCK', ((1, 3), (0, 4))),  # LM - LAKY
        (900 - 18.01056027, b'y', 2, 1, 2, b'MCK_H2Ox1', ((1, 3), (0, 4))),  # LM - LAKY
    ], dtype=mock_frag_dtype)

    out_frags = include_losses(in_fragments, np.array([0, 1]), ctx)

    expected_fragments.sort(order=['mz'])

    np.testing.assert_almost_equal(out_frags['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(out_frags[['ion_type', 'idx', 'charge', 'nlosses', 'loss', 'ranges']],
                            expected_fragments[['ion_type', 'idx', 'charge', 'nlosses', 'loss',
                                                'ranges']])


def test_include_losses_X():
    # columns needed: term, LN, nlosses, ranges, loss, mz
    # as include losses is kind of a thin wrapper around create_loss_fragments this is
    # functioning as a test for create_loss_fragments  as well

    water_loss = Loss(name='H2O', composition="H2O1", specificity=['X'])

    mock_frag_dtype = np.dtype([
        ('mz', np.float64), ('ion_type', 'S1'), ('idx', np.uint8),
        ('charge', np.uint8),
        ('nlosses', np.uint8),
        ('loss', 'S25'),
        ('ranges', np.uint8, (2, 2))
    ])

    ctx = MockContext(
        Config(crosslinker=Crosslinker.BS3, ms2_tol='1ppm',
               fragmentation=FragmentationConfig(losses=[water_loss], max_nloss=4))
    )
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))

    in_fragments = np.array([
        (900, b'y', 2, 1, 1, b'MCK', ((0, 0), (0, 4))),  # LAKY
    ], dtype=mock_frag_dtype)

    expected_fragments = np.array([
        # mock losses
        (900, b'y', 2, 1, 1, b'MCK', ((0, 0), (0, 4))),  # LAKY
        (900 - 18.01056027, b'y', 2, 1, 2, b'MCK_H2Ox1', ((0, 0), (0, 4))),  # LAKY
        (900 - 2*18.01056027, b'y', 2, 1, 3, b'MCK_H2Ox2', ((0, 0), (0, 4))),  # LAKY
        (900 - 3*18.01056027, b'y', 2, 1, 4, b'MCK_H2Ox3', ((0, 0), (0, 4))),  # LAKY
    ], dtype=mock_frag_dtype)

    out_frags = include_losses(in_fragments, np.array([0, 1]), ctx)

    expected_fragments.sort(order=['mz'])

    np.testing.assert_almost_equal(out_frags['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(out_frags[['ion_type', 'idx', 'charge', 'nlosses', 'loss', 'ranges']],
                            expected_fragments[['ion_type', 'idx', 'charge', 'nlosses', 'loss',
                                                'ranges']])


def test_include_losses_no_loss():
    # columns needed: term, LN, nlosses, ranges, loss, mz
    # as include losses is kind of a thin wrapper around create_loss_fragments this is
    # functioning as a test for create_loss_fragments  as well

    water_loss = Loss(name='H2O', composition="H2O1", specificity=['N'])

    mock_frag_dtype = np.dtype([
        ('mz', np.float64), ('ion_type', '<U1'), ('idx', np.uint8),
        ('charge', np.uint8),
        ('nlosses', np.uint8),
        ('loss', '<U25'),
        ('ranges', np.uint8, (2, 2))
    ])

    ctx = MockContext(
        Config(crosslinker=Crosslinker.BS3, ms2_tol='1ppm',
               fragmentation=FragmentationConfig(losses=[water_loss], max_nloss=4))
    )
    ctx.setup_peptide_db(np.array([b"KLM", b"LAKY"]))

    in_fragments = np.array([
        (900, 'y', 2, 1, 1, 'MCK', ((0, 0), (0, 4))),  # LAKY
    ], dtype=mock_frag_dtype)

    expected_fragments = np.array([
        # mock losses
        (900, 'y', 2, 1, 1, 'MCK', ((0, 0), (0, 4))),  # LAKY
    ], dtype=mock_frag_dtype)

    out_frags = include_losses(in_fragments, np.array([0, 1]), ctx)

    expected_fragments.sort(order=['mz'])

    np.testing.assert_almost_equal(out_frags['mz'], expected_fragments['mz'], decimal=4)
    np.testing.assert_equal(out_frags[['ion_type', 'idx', 'charge', 'nlosses', 'loss', 'ranges']],
                            expected_fragments[['ion_type', 'idx', 'charge', 'nlosses', 'loss',
                                                'ranges']])
