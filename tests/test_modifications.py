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

from xicommon.modifications import *
from xicommon.config import Config, ModificationConfig, Modification, Crosslinker
from xicommon.mock_context import MockContext
from numpy.testing import assert_array_equal
from xicommon import dtypes


def test_var_mods():
    """Test variable side-chain modifications."""
    var_mod = Modification(name='v', specificity=['C'], type='variable', mass=1)
    config = Config(modification=ModificationConfig(modifications=[var_mod]))
    ctx = MockContext(config)

    peptides = np.array([b'ACCA'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([(0, 0, 0, False, False, False, False)], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)

    assert len(mod_peptides) == 4

    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0, 0], 0, False, 0),  # ACCA
        (0, [0, 0, 0, 1, 0, 0], 0, False, 1),  # AvCCA
        (0, [0, 0, 0, 0, 1, 0], 0, False, 1),  # ACvCA
        (0, [0, 0, 0, 1, 1, 0], 0, False, 2),  # AvCvCA
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)


def test_var_mods_multimod():
    """Test variable side-chain modifications."""
    var_mod1 = Modification(name='x', specificity=['C'], type='variable', mass=1)
    var_mod2 = Modification(name='y', specificity=['C', 'K'], type='variable', mass=1)
    config = Config(modification=ModificationConfig(modifications=[var_mod1, var_mod2]))
    ctx = MockContext(config)

    peptides = np.array([b'AKCKA'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)

    assert len(mod_peptides) == 10

    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0, 0, 0], 0, False, 0),  # AKCKA
        (0, [0, 0, 0, 0, 1, 0, 0], 0, False, 1),  # AKxCKA
        (0, [0, 0, 0, 2, 0, 0, 0], 0, False, 1),  # AyKCKA
        (0, [0, 0, 0, 0, 2, 0, 0], 0, False, 1),  # AKyCKA
        (0, [0, 0, 0, 0, 0, 2, 0], 0, False, 1),  # AKCyKA
        (0, [0, 0, 0, 2, 1, 0, 0], 0, False, 2),  # AyKxCKA
        (0, [0, 0, 0, 2, 2, 0, 0], 0, False, 2),  # AyKyCKA
        (0, [0, 0, 0, 0, 1, 2, 0], 0, False, 2),  # AKxCyKA
        (0, [0, 0, 0, 2, 0, 2, 0], 0, False, 2),  # AyKCyKA
        (0, [0, 0, 0, 0, 2, 2, 0], 0, False, 2),  # AKyCyKA
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)


def test_var_mods_multiple_peptides():
    """Test multiple variable protein modifications"""
    var_mod = Modification(name='v', specificity=['C', 'A'], type='variable', level='protein',
                           mass=1)
    config = Config(modification=ModificationConfig(modifications=[var_mod]))
    ctx = MockContext(config)

    peptides = np.array([b'ACDK', b'ACEK', b'ACK', b'ACHK'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, True, False, False),
        (1, 0, 0, False, True, False, False),
        (2, 0, 0, True, False, False, False),
        (3, 0, 0, True, True, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)

    expected_mod_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0, 0], 0, False, 0),  # ACDK
        (1, [0, 0, 0, 0, 0, 0], 1, False, 0),  # ACEK
        (2, [0, 0, 0, 0, 0, 0], 2, False, 0),  # ACK
        (3, [0, 0, 0, 0, 0, 0], 3, False, 0),  # ACHK

        (0, [0, 0, 1, 0, 0, 0], 0, False, 1),  # vACDK
        (1, [0, 0, 1, 0, 0, 0], 1, False, 1),  # vACEK
        (2, [0, 0, 1, 0, 0, 0], 2, False, 1),  # vACK
        (3, [0, 0, 1, 0, 0, 0], 3, False, 1),  # vACHK
        (0, [0, 0, 0, 1, 0, 0], 0, False, 1),  # AvCDK
        (1, [0, 0, 0, 1, 0, 0], 1, False, 1),  # AvCEK
        (2, [0, 0, 0, 1, 0, 0], 2, False, 1),  # AvCK
        (3, [0, 0, 0, 1, 0, 0], 3, False, 1),  # AvCHK

        (0, [0, 0, 1, 1, 0, 0], 0, False, 2),  # vAvCDK
        (1, [0, 0, 1, 1, 0, 0], 1, False, 2),  # vAvCEK
        (2, [0, 0, 1, 1, 0, 0], 2, False, 2),  # vAvCK
        (3, [0, 0, 1, 1, 0, 0], 3, False, 2),  # vAvCHK
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_mod_peptides)


def test_peptide_linear_mods():
    """Test linear peptide modification"""
    lin_mod = Modification(name='v', specificity=['C', 'A'], type='linear', level='peptide',
                           mass=1)
    config = Config(modification=ModificationConfig(modifications=[lin_mod]))
    ctx = MockContext(config)

    peptides = np.array([b'ACDK', b'ACEK', b'ACK', b'ACHK'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, True, False, False),
        (1, 0, 0, False, True, False, False),
        (2, 0, 0, True, False, False, False),
        (3, 0, 0, True, True, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)

    expected_mod_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0, 0], 0, False, 0),  # ACDK
        (1, [0, 0, 0, 0, 0, 0], 1, False, 0),  # ACEK
        (2, [0, 0, 0, 0, 0, 0], 2, False, 0),  # ACK
        (3, [0, 0, 0, 0, 0, 0], 3, False, 0),  # ACHK

        (0, [0, 0, 1, 0, 0, 0], 0, True, 1),  # vACDK
        (1, [0, 0, 1, 0, 0, 0], 1, True, 1),  # vACEK
        (2, [0, 0, 1, 0, 0, 0], 2, True, 1),  # vACK
        (3, [0, 0, 1, 0, 0, 0], 3, True, 1),  # vACHK
        (0, [0, 0, 0, 1, 0, 0], 0, True, 1),  # AvCDK
        (1, [0, 0, 0, 1, 0, 0], 1, True, 1),  # AvCEK
        (2, [0, 0, 0, 1, 0, 0], 2, True, 1),  # AvCK
        (3, [0, 0, 0, 1, 0, 0], 3, True, 1),  # AvCHK

        (0, [0, 0, 1, 1, 0, 0], 0, True, 2),  # vAvCDK
        (1, [0, 0, 1, 1, 0, 0], 1, True, 2),  # vAvCEK
        (2, [0, 0, 1, 1, 0, 0], 2, True, 2),  # vAvCK
        (3, [0, 0, 1, 1, 0, 0], 3, True, 2),  # vAvCHK
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_mod_peptides)


def test_var_nterm_mods():
    """Test peptide and protein level variable n-terminal modifications."""
    pep_residue_mod = Modification(name='pe1', specificity=['K'], type='variable',
                                   level='peptide', mass=1)
    pep_n_term_mod = Modification(name='pe2-', specificity=['X'], type='variable',
                                  level='peptide', mass=1)
    prot_residue_mod = Modification(name='pr1', specificity=['K'], type='variable',
                                    level='protein', mass=1)
    prot_n_term_mod = Modification(name='pr2-', specificity=['X'], type='variable',
                                   level='protein', mass=1)

    test_mods = [
        pep_residue_mod, pep_n_term_mod,
        prot_residue_mod, prot_n_term_mod]
    config = Config(modification=ModificationConfig(modifications=test_mods))
    ctx = MockContext(config)

    peptides = np.array([b'AAK', b'CC'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # 1. Peptides are not protein N-terminal so only peptide N-terminal mod should be applied
    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, False),
        (1, 0, 0, False, True, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)
    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0], 0, False, 0),  # AAK
        (1, [0, 0, 0, 0, 0], 1, False, 0),  # CC
        (0, [0, 0, 0, 0, 3], 0, False, 1),  # AApr1K
        (0, [0, 0, 0, 0, 1], 0, False, 1),  # AApe1K
        (0, [2, 0, 0, 0, 0], 0, False, 1),  # pe2-AAK
        (0, [2, 0, 0, 0, 3], 0, False, 2),  # pe2-AApr1K
        (1, [2, 0, 0, 0, 0], 1, False, 1),  # pe2-CC
        (0, [2, 0, 0, 0, 1], 0, False, 2),  # pe2-AApe1K
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)

    # 2. Peptides are protein N-terminal so peptide and protein n-terminal mods should be applied
    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, True, False, False, False),
        (1, 0, 0, True, True, False, False),
    ], dtype=dtypes.site_info_table)
    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)
    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0], 0, False, 0),  # AAK
        (1, [0, 0, 0, 0, 0], 1, False, 0),  # CC
        (0, [0, 0, 0, 0, 3], 0, False, 1),  # AApr1K
        (0, [4, 0, 0, 0, 0], 0, False, 1),  # pr2-AAK
        (1, [4, 0, 0, 0, 0], 1, False, 1),  # pr2-CC
        (0, [4, 0, 0, 0, 3], 0, False, 2),  # pr2-AApr1K
        (0, [0, 0, 0, 0, 1], 0, False, 1),  # AApe1K
        (0, [4, 0, 0, 0, 1], 0, False, 2),  # pr2-AApe1K
        (0, [2, 0, 0, 0, 0], 0, False, 1),  # pe2-AAK
        (0, [2, 0, 0, 0, 3], 0, False, 2),  # pe2-AApr1K
        (1, [2, 0, 0, 0, 0], 1, False, 1),  # pe2-CC
        (0, [2, 0, 0, 0, 1], 0, False, 2),  # pe2-AApe1K
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)


def test_var_cterm_mods():
    """Test peptide and protein level variable c-terminal modifications."""
    pep_residue_mod = Modification(name='pe1', specificity=['K'], type='variable',
                                   level='peptide', mass=1)
    pep_c_term_mod = Modification(name='-pe2', specificity=['X'], type='variable',
                                  level='peptide', mass=1)
    prot_residue_mod = Modification(name='pr1', specificity=['K'], type='variable',
                                    level='protein', mass=1)
    prot_c_term_mod = Modification(name='-pr2', specificity=['X'], type='variable',
                                   level='protein', mass=1)

    test_mods = [pep_residue_mod, pep_c_term_mod, prot_residue_mod, prot_c_term_mod]
    config = Config(modification=ModificationConfig(modifications=test_mods))
    ctx = MockContext(config)

    peptides = np.array([b'AAK', b'CC'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # 1. Peptides are not protein C-terminal so only peptide N-terminal mod should be applied
    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, False),
        (1, 0, 0, True, False, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)
    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0], 0, False, 0),  # AAK
        (1, [0, 0, 0, 0, 0], 1, False, 0),  # CC
        (0, [0, 0, 0, 0, 3], 0, False, 1),  # AApr1K
        (0, [0, 0, 0, 0, 1], 0, False, 1),  # AApe1K
        (0, [0, 2, 0, 0, 0], 0, False, 1),  # AAK-pe2
        (0, [0, 2, 0, 0, 3], 0, False, 2),  # AApr1K-pe2
        (1, [0, 2, 0, 0, 0], 1, False, 1),  # CC-pe2
        (0, [0, 2, 0, 0, 1], 0, False, 2),  # AApe1K-pe2
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)

    # 2. Peptides are protein C-terminal so peptide and protein n-terminal mods should be applied
    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, True, False, False),
        (1, 0, 0, True, True, False, False),
    ], dtype=dtypes.site_info_table)
    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)
    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only # pep seq in modX syntax
        (0, [0, 0, 0, 0, 0], 0, False, 0),  # AAK
        (1, [0, 0, 0, 0, 0], 1, False, 0),  # CC
        (0, [0, 0, 0, 0, 3], 0, False, 1),  # AApr1K
        (0, [0, 4, 0, 0, 0], 0, False, 1),  # AAK-pr2
        (1, [0, 4, 0, 0, 0], 1, False, 1),  # CC-pr2
        (0, [0, 4, 0, 0, 3], 0, False, 2),  # AApr1K-pr2
        (0, [0, 0, 0, 0, 1], 0, False, 1),  # AApe1K
        (0, [0, 4, 0, 0, 1], 0, False, 2),  # AApe1K-pr2
        (0, [0, 2, 0, 0, 0], 0, False, 1),  # AAK-pe2
        (0, [0, 2, 0, 0, 3], 0, False, 2),  # AApr1K-pe2
        (1, [0, 2, 0, 0, 0], 1, False, 1),  # CC-pe2
        (0, [0, 2, 0, 0, 1], 0, False, 2),  # AApe1K-pe2
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)


def test_max_var_mods():
    """Test maximum number of variable modifications."""
    prot_mod1 = Modification(name='pr1', specificity=['A'], type='variable', level='protein',
                             mass=1)
    prot_mod2 = Modification(name='pr2', specificity=['C'], type='variable', level='protein',
                             mass=1)
    pep_mod1 = Modification(name='pe1-', specificity=['X'], type='variable', level='peptide',
                            mass=1)
    pep_mod2 = Modification(name='-pe2', specificity=['X'], type='variable', level='peptide',
                            mass=1)
    test_mods = [prot_mod1, prot_mod2, pep_mod1, pep_mod2]
    config = Config(modification=ModificationConfig(
        modifications=test_mods, max_var_protein_mods=2, max_var_peptide_mods=1,
        max_modified_peps=100))
    ctx = MockContext(config)

    peptides = np.array([b'ACCA'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptide_arrs = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptide_arrs, site_info)

    mod_seq_strs = modified_sequence_strings(ctx.unmodified_peptide_sequences, mod_peptides, config)
    # maximum 2 protein level variable modifications
    assert max([p.count(b'pr') for p in mod_seq_strs]) == 2
    # maximum 1 peptide level variable modification
    assert max([p.count(b'pe') for p in mod_seq_strs]) == 1

    # make sure there could be more with higher limits
    config = Config(modification=ModificationConfig(
        modifications=test_mods, max_var_protein_mods=10, max_var_peptide_mods=10,
        max_modified_peps=100))
    ctx = MockContext(config)
    ctx.setup_peptide_db(peptides)
    peptide_arrs = ctx.modified_peptides
    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptide_arrs, site_info)
    mod_seq_strs = modified_sequence_strings(ctx.unmodified_peptide_sequences, mod_peptides, config)
    assert max([p.count(b'pr') for p in mod_seq_strs]) > 2
    assert max([p.count(b'pe') for p in mod_seq_strs]) > 1


def test_max_modified_peps():
    """Test maximum number of modified peptides."""
    prot_mod1 = Modification(name='pr1', specificity=['A'], type='variable', level='protein',
                             mass=1)
    prot_mod2 = Modification(name='pr2', specificity=['C'], type='variable', level='protein',
                             mass=1)
    pep_mod1 = Modification(name='pe1-', specificity=['X'], type='variable', level='peptide',
                            mass=1)
    pep_mod2 = Modification(name='-pe2', specificity=['X'], type='variable', level='peptide',
                            mass=1)
    test_mods = [prot_mod1, prot_mod2, pep_mod1, pep_mod2]
    config = Config(modification=ModificationConfig(modifications=test_mods, max_modified_peps=4))
    ctx = MockContext(config)

    peptides = np.array([b'ACCA', b'LCCA'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptide_arrs = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, False),
        (1, 0, 0, False, False, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptide_arrs, site_info)

    # make sure we only get 10 (4+1 unmodified * 2 peptides)
    assert len(mod_peptides) == 10

    # check that we always get the same 10
    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        # unmodified
        (0, [0, 0, 0, 0, 0, 0], 0, False, 0),  # ACCA
        (1, [0, 0, 0, 0, 0, 0], 1, False, 0),  # LCCA

        # 1 modification
        (0, [0, 0, 1, 0, 0, 0], 0, False, 1),  # pr1ACCA
        (0, [0, 0, 0, 0, 0, 1], 0, False, 1),  # ACCpr1A
        (1, [0, 0, 0, 0, 0, 1], 1, False, 1),  # LCCpr1A
        (0, [0, 0, 0, 2, 0, 0], 0, False, 1),  # Apr2CCA
        (1, [0, 0, 0, 2, 0, 0], 1, False, 1),  # Lpr2CCA
        (0, [0, 0, 0, 0, 2, 0], 0, False, 1),  # ACpr2CA
        (1, [0, 0, 0, 0, 2, 0], 1, False, 1),  # LCpr2CA

        # 2 modifications
        (1, [0, 0, 0, 2, 0, 1], 1, False, 2),  # Lpr2CCpr1A

    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)

    # make sure there could be more with a higher limit
    config = Config(modification=ModificationConfig(modifications=test_mods, max_modified_peps=20))
    ctx = MockContext(config)
    ctx.setup_peptide_db(peptides)
    peptide_arrs = ctx.modified_peptides
    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptide_arrs, site_info)
    assert len(mod_peptides) > 10


def test_max_modified_peps_consistency():
    """
    Test that we always get the same variable modifications back for two variable modifications
    that have overlapping specificity.
    """
    prot_mod1 = Modification(name='bs3oh', specificity=['K'], type='variable', level='protein',
                             mass=1)
    prot_mod2 = Modification(name='bs3nh2', specificity=['K'], type='variable', level='protein',
                             mass=1)
    test_mods = [prot_mod1, prot_mod2]
    config = Config(modification=ModificationConfig(modifications=test_mods, max_modified_peps=5))
    ctx = MockContext(config)

    peptides = np.array([b'AKKAK'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptide_arrs = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptide_arrs, site_info)
    # make sure we only get 6 (5+1 unmodified)
    assert len(mod_peptides) == 6

    # check that we always get the same 6
    # ordered by number of modifications then modification index then residue position (from N to C)
    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only, var_mod_count  # modX seq
        (0, [0, 0, 0, 0, 0, 0, 0], 0, False, 0),  # AKKAK
        (0, [0, 0, 0, 1, 0, 0, 0], 0, False, 1),  # Abs3ohKKAK
        (0, [0, 0, 0, 0, 1, 0, 0], 0, False, 1),  # AKbs3ohKAK
        (0, [0, 0, 0, 0, 0, 0, 1], 0, False, 1),  # AKKAbs3ohK
        (0, [0, 0, 0, 2, 0, 0, 0], 0, False, 1),  # Abs3nh2KKAK
        (0, [0, 0, 0, 0, 2, 0, 0], 0, False, 1),  # AKbs3nh2KAK
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)


def test_modified_sequence_strings():
    """Test modified sequence strings with modX and Xmod syntax."""
    # define test modifications
    m1 = Modification(name='a', specificity=['K'], type='variable', composition="H1")
    m2 = Modification(name='b', specificity=['C'], type='variable', composition="H1")

    # test modX syntax
    cfg = Config(modification=ModificationConfig(modifications=[m1, m2]))
    ctx = MockContext(cfg)
    peptides = np.array([b'AAaKAA', b'AAKAAbC'])
    ctx.setup_peptide_db(peptides)

    mod_seqs = modified_sequence_strings(ctx.peptide_db.unmodified_sequences,
                                         ctx.peptide_db.peptides, cfg, mod_position='modX')
    # modX syntax so mod_seqs should be the same as input
    expected = peptides
    assert_array_equal(expected, mod_seqs)

    # test Xmod syntax
    mod_seqs = modified_sequence_strings(ctx.peptide_db.unmodified_sequences,
                                         ctx.peptide_db.peptides, cfg, mod_position='Xmod')

    expected = np.array([b'AAKaAA', b'AAKAACb'])
    assert_array_equal(expected, mod_seqs)


def test_linear_only():
    """Test that 'linear_only' (non-crosslinkable) flag is set correctly on modified peptides."""
    var_mod = Modification(name='v', specificity=['K'], type='variable', mass=1)
    lin_mod = Modification(name='l', specificity=['C'], type='linear', mass=1)
    config = Config(
        modification=ModificationConfig(modifications=[var_mod, lin_mod]),
        crosslinker=Crosslinker.BS3)
    ctx = MockContext(config)

    peptides = np.array([b'AKCA', b'CAA'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, False),
        (1, 0, 0, False, False, False, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)

    assert len(mod_peptides) == 6

    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only # pep seq in modX syntax
        (0, [0, 0, 0, 0, 0, 0], 0, False, 0),  # AKCA
        (1, [0, 0, 0, 0, 0, 0], 1, False, 0),  # CAA
        (0, [0, 0, 0, 1, 0, 0], 0, False, 1),  # AvKCA
        (1, [0, 0, 2, 0, 0, 0], 1, True, 1),  # lCAA
        (0, [0, 0, 0, 0, 2, 0], 0, True, 1),  # AKlCA
        (0, [0, 0, 0, 1, 2, 0], 0, True, 2),  # AvKlCA
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)


def test_digestibility():
    """
    Test digestibility check (nterm_aa_block and cterm_aa_block)
    """
    var_mods = [
        Modification(name='k', specificity=['K'], type='variable', mass=1),
        Modification(name='c', specificity=['C'], type='variable', mass=1),
    ]
    config = Config(modification=ModificationConfig(modifications=var_mods),
                    crosslinker=Crosslinker.BS3)
    ctx = MockContext(config)

    peptides = np.array([b'ACAK', b'CKA'])
    # setup peptide db
    ctx.setup_peptide_db(peptides)
    # ctx.modified peptides now holds the peptides in array form
    peptides = ctx.modified_peptides

    # sequence_index, sites_first_idx, sites_last_idx, nterm, cterm, nterm_aa_block, cterm_aa_block
    site_info = np.array([
        (0, 0, 0, False, False, False, True),
        (1, 0, 0, False, False, True, False),
    ], dtype=dtypes.site_info_table)

    my_modifier = Modifier(ctx)
    mod_peptides = my_modifier.apply_remaining_mods(peptides, site_info)

    expected_peptides = np.array([
        # sequence index, modification array, site index, linear_only # pep seq in modX syntax
        (0, [0, 0, 0, 0, 0, 0], 0, False, 0),  # ACAK
        (1, [0, 0, 0, 0, 0, 0], 1, False, 0),  # CKA
        (1, [0, 0, 0, 1, 0, 0], 1, False, 1),  # CkKA
        (0, [0, 0, 0, 2, 0, 0], 0, False, 1),  # AcCAK
    ], dtype=mod_peptides.dtype)
    assert_array_equal(mod_peptides, expected_peptides)
