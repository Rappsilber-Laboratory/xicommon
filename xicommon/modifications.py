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

"""Module for handling protein/peptide modifications."""
from pyteomics import parser
import numpy as np
import re
from xicommon.xi_logging import ProgressBar, log
from xicommon.utils import concatenated_ranges


class Modifier:
    """Class that applies modifications on amino acid sequences."""

    has_nterm_mod = re.compile(b'(^[^A-Z]-.*)')
    has_cterm_mod = re.compile(b'(.*-[^A-Z]*$)')

    def __init__(self, context):
        """Initialize the Modifier."""
        self.context = context
        self.config = context.config.modification

        # set the index of the modification for the array form (0 is reserved for unmodified)
        for mod_idx, mod in enumerate(self.config.modifications):
            mod.index = mod_idx + 1

        self.protein_var_mods = [
            m for m in self.config.modifications if m.type == 'variable' and m.level == 'protein']
        self.protein_fix_mods = self._create_protein_fix_mods()
        self.protein_lin_mods = [
            m for m in self.config.modifications if m.type == 'linear' and m.level == 'protein']
        self.ms3_stub_mods = [
            m for m in self.config.modifications if m.type == 'ms3stub' and m.level == 'protein']
        self.peptide_var_mods = [
            m for m in self.config.modifications if m.type == 'variable' and m.level == 'peptide']
        self.peptide_fix_mods = [
            m for m in self.config.modifications if m.type == 'fixed' and m.level == 'peptide']
        self.peptide_lin_mods = [
            m for m in self.config.modifications if m.type == 'linear' and m.level == 'peptide']

        # modX labels for all amino acids, unmodified termini and all defined modifications
        self.labels = parser.std_labels + ['U', 'O', 'X'] + \
            [m.name for m in self.config.modifications]

        self.mod_pep_count = None

    def apply_protein_fixed_mods(self, sequences):
        """
        Apply fixed protein level modifications on an array of sequences.

        :param sequences: (ndarray bytes) amino acid sequences
        :return: (ndarray bytes) modified amino acid sequences
        """
        if len(self.protein_fix_mods) == 0:
            return sequences

        return_sequences = []
        bar = ProgressBar("Applying %d fixed protein modifications" % len(self.protein_fix_mods),
                          len(sequences))
        for sequence in sequences:
            mod_gen = parser.isoforms(sequence.decode('ascii'), fixed_mods=self.protein_fix_mods,
                                      labels=self.labels)
            return_sequences.extend(list(mod_gen))
            bar.next()
        bar.finish()

        return np.array(return_sequences, bytes)

    def apply_remaining_mods(self, peptides, site_info_table):
        """
        Apply variable & fixed peptide level and variable protein level modifications.

        :param peptides: (struct ndarray)
            'sequence_index': (intp) index of the unmodified base sequence
            'modifications': (uint8) modification array
        :param site_info_table: (struct ndarray)
            sequence_index (intp) index into the base sequence
            sites_first_idx (intp) first index into peptide sites (Searcher.sites)
            sites_last_idx (intp) last index into peptide sites (Searcher.sites)
            nterm (bool_) if the peptide occurs n-terminal (on any protein)
            cterm (bool_) if the peptide occurs c-terminal (on any protein)
            nterm_aa_block (bool_) nterm aa is blocked on all proteins
            cterm_aa_block (bool_) cterm aa is blocked on all proteins
        :return: Modified and unmodified versions of the input peptides
        :rtype: (struct ndarray)
            sequence_index: (intp) index of the unmodified base sequence
            modifications: (uint8) modification array
            site_info_idx: (intp) index into the sites info table
            linear_only: (bool) linear only peptide (by modification definition)
            var_mod_count: (uint8) variable modification count
        """
        # initialise the modified peptide counter
        self.mod_pep_count = np.zeros(len(peptides), np.intp)

        seq_idx = peptides['sequence_index']
        mod_arrs = peptides['modifications']
        site_info_idx = np.arange(peptides.size)
        lin_arr = np.zeros(len(seq_idx), np.bool_)

        # 1. Apply ms3 stub modifications
        if len(self.ms3_stub_mods) > 0:
            log(f"Applying {len(self.ms3_stub_mods)} cleavable crosslinker stub modifications "
                f"(MS3 search)")
            seq_idx, mod_arrs, site_info_idx, lin_arr = \
                self.apply_var_modifications(
                    seq_idx, mod_arrs, site_info_idx, lin_arr, site_info_table,
                    self.ms3_stub_mods, 1, 'protein', True)

        # 2. Apply protein variable modifications
        if len(self.protein_var_mods) > 0:
            log(f"Applying {len(self.protein_var_mods)} protein variable modifications")
            seq_idx, mod_arrs, site_info_idx, lin_arr = \
                self.apply_var_modifications(
                    seq_idx, mod_arrs, site_info_idx, lin_arr, site_info_table,
                    self.protein_var_mods, self.config.max_var_protein_mods, 'protein', False)

        # 3. Peptide variable modifications
        if len(self.peptide_var_mods) > 0:
            log(f"Applying {len(self.peptide_var_mods)} peptide variable modifications")
            seq_idx, mod_arrs, site_info_idx, lin_arr = \
                self.apply_var_modifications(
                    seq_idx, mod_arrs, site_info_idx, lin_arr, site_info_table,
                    self.peptide_var_mods, self.config.max_var_peptide_mods, 'peptide', False)

        # 4. Protein linear only modifications
        if len(self.protein_lin_mods) > 0:
            log(f"Applying {len(self.protein_lin_mods)} protein linear-only modifications")
            seq_idx, mod_arrs, site_info_idx, lin_arr = \
                self.apply_var_modifications(
                    seq_idx, mod_arrs, site_info_idx, lin_arr, site_info_table,
                    self.protein_lin_mods, self.config.max_linear_protein_mods, 'protein', True)

        # 5. Peptide linear only modifications
        if len(self.peptide_lin_mods) > 0:
            log(f"Applying {len(self.peptide_lin_mods)} peptide linear-only modifications")
            seq_idx, mod_arrs, site_info_idx, lin_arr = \
                self.apply_var_modifications(
                    seq_idx, mod_arrs, site_info_idx, lin_arr, site_info_table,
                    self.peptide_lin_mods, self.config.max_linear_peptide_mods, 'peptide', True)

        # 6. Peptide fixed modifications
        if len(self.peptide_fix_mods) > 0:
            log(f"Applying {len(self.peptide_fix_mods)} peptide fixed modifications")
            mod_arrs = self.apply_fixed_modifications(
                seq_idx, mod_arrs, site_info_idx, site_info_table, self.peptide_fix_mods, 'peptide')

        output_dtype = [
            ('sequence_index', site_info_table['sequence_index'].dtype),
            ('modifications', peptides['modifications'].dtype,
             (peptides['modifications'].shape[1])),
            ('site_info_idx', np.intp),
            ('linear_only', np.bool_),
            ('var_mod_count', np.uint8)
        ]
        output = np.empty(len(seq_idx), dtype=output_dtype)
        output['sequence_index'] = seq_idx
        output['modifications'] = mod_arrs
        output['site_info_idx'] = site_info_idx
        output['linear_only'] = lin_arr

        # count the number of variable modifications
        var_mod_idxs = np.array([m.index for m in self.protein_var_mods + self.protein_lin_mods
                                 + self.peptide_var_mods + self.peptide_lin_mods])
        output['var_mod_count'] = np.any(
            mod_arrs == var_mod_idxs.reshape(-1, 1, 1), axis=0).sum(axis=1)

        return output

    @classmethod
    def _format_to_pyteomics_syntax(cls, modifications):
        """
        Reformat list of modifications to pyteomics.parser dict syntax.

        Replace 'X' in specificity with (bool) True (as expected by pyteomics for unspecific mod).
        """
        return {m.name: (True if 'X' in m.specificity else m.specificity)
                for m in modifications}

    def _create_protein_fix_mods(self):
        """Return the protein fixed modification in the dict syntax used by pyteomics.parser."""
        fix_mods = [m for m in self.config.modifications if m.type == 'fixed'
                    and m.level == 'protein']
        return self._format_to_pyteomics_syntax(fix_mods)

    def create_matches_array(self, seq_idx_arr, mod_array, site_info, modifications,
                             level='protein'):
        """
        Match each peptide sequences to modification specificities.

        :param seq_idx_arr: (ndarray, intp) base sequence indices
        :param mod_array: (ndarray, uint8) modification array
        :param site_info: (struct ndarray)
            sequence_index (intp) index into the base sequence
            sites_first_idx (intp) first index into peptide sites (Searcher.sites)
            sites_last_idx (intp) last index into peptide sites (Searcher.sites)
            nterm (bool_) if the peptide occurs n-terminal (on any protein)
            cterm (bool_) if the peptide occurs c-terminal (on any protein)
            nterm_aa_block (bool_) nterm aa is blocked on all proteins
            cterm_aa_block (bool_) cterm aa is blocked on all proteins
        :param modifications: (list Modification) list of modifications to match
        :param level: (str) level to match on (protein or peptide). Affects how terminal
            modifications are matched
        :return: matches array
        :rtype: ndarray bool ndim=3
            dim 1: peptides
            dim 2: amino acids
            dim 3: modifications
        """
        # get peptide base sequences as uint8
        sequences = self.context.unmodified_peptide_sequences[seq_idx_arr].view(
            np.uint8).reshape(len(seq_idx_arr), -1)

        # create mod aa specificity array
        aa_spec_list = [m.ord_aa_specificity for m in modifications]

        aa_spec = np.full([len(aa_spec_list), len(max(aa_spec_list, key=lambda x: len(x)))], -1)
        for i, aa in enumerate(aa_spec_list):
            aa_spec[i][0:len(aa)] = aa

        # match the sequences to the aa specificities
        # creates a 3D array with:
        #     dim 1: peptides
        #     dim 2: amino acids
        #     dim 3: modifications
        aa_spec_matches = np.any((sequences.reshape(-1, sequences.shape[1], 1, 1) == aa_spec),
                                 axis=3)

        # handle wildcard (X) specificity
        wildcard = np.any(aa_spec == 88, axis=1)
        aa_spec_matches = aa_spec_matches | wildcard

        # terminal modifications
        nterm_matches = [m.nterm_mod for m in modifications]
        cterm_matches = [m.cterm_mod for m in modifications]
        term_matches = np.zeros((aa_spec_matches.shape[0], 2, aa_spec_matches.shape[2]),
                                dtype=np.bool_)
        if level == 'protein':
            # for protein level mods set only the peptides that are protein terminal to match
            term_matches[site_info['nterm'], 0, :] = nterm_matches
            term_matches[site_info['cterm'], 1, :] = cterm_matches
        else:
            term_matches[:, 0, :] = nterm_matches
            term_matches[:, 1, :] = cterm_matches
        # reset all the side-chain specificities for these mods to False
        aa_spec_matches[:, :, nterm_matches] = False
        aa_spec_matches[:, :, cterm_matches] = False

        # create the matches array by combining the aa_spec with the term_spec
        spec_matches = np.empty((aa_spec_matches.shape[0], aa_spec_matches.shape[1] + 2,
                                 aa_spec_matches.shape[2]), dtype=np.bool_)
        spec_matches[:, 2:, :] = aa_spec_matches
        spec_matches[:, :2, :] = term_matches

        # skip already modified aa
        already_modified = mod_array.astype(np.bool_)
        spec_matches = spec_matches & ~already_modified.reshape(
            already_modified.shape[0], already_modified.shape[1], -1)
        # set nterm and cterm block
        if level == 'protein':
            last_aa_idx = sequences.astype(np.bool_).sum(axis=1) + 1
            nterm_block = site_info['nterm_aa_block']
            cterm_block = site_info['cterm_aa_block']
            spec_matches[nterm_block, 2, :] = False
            spec_matches[cterm_block, last_aa_idx[cterm_block], :] = False
        # set matches that belong to seq_idx that have reached max_modified_peps to False
        spec_matches[self.mod_pep_count[seq_idx_arr] >= self.config.max_modified_peps] = False

        return spec_matches

    def apply_fixed_modifications(self, seq_indices, mod_arrs, site_info_idx,
                                  site_info_table, modifications, level):
        """
        Apply a set of fixed modifications on an array of peptides.

        :param seq_indices: (ndarray int64) array of base sequence indices
        :param mod_arrs: (ndarray uint8) input modification arrays
        :param site_info_idx: (ndarray int64) indices into the sites idx table
        :param site_info_table: (struct ndarray)
            sequence_index (intp) index into the base sequence
            sites_first_idx (intp) first index into peptide sites (Searcher.sites)
            sites_last_idx (intp) last index into peptide sites (Searcher.sites)
            nterm (bool_) if the peptide occurs n-terminal (on any protein)
            cterm (bool_) if the peptide occurs c-terminal (on any protein)
            nterm_aa_block (bool_) nterm aa is blocked on all proteins
            cterm_aa_block (bool_) cterm aa is blocked on all proteins
        :param modifications: (list Modification) list of modifications to apply
        :param level: (str) 'protein' or 'peptide' level modifications

        return: modification arrays of modified peptides
        rtype: ndarray uint8
        """
        # find the matches between peptides an modifications
        matches = self.create_matches_array(
            seq_indices, mod_arrs, site_info_table[site_info_idx], modifications, level)

        mod_idx_arr = [m.index for m in modifications]
        # loop over modification dimension (2)
        for i in range(matches.shape[2]):
            mod_matches = matches[:, :, i]
            mod_arrs[mod_matches] = mod_idx_arr[i]

        return mod_arrs

    def apply_var_modifications(self, seq_indices, mod_arrs, site_info_idx, lin_arr,
                                site_info_table, modifications, max_mods, level, linear_only):
        """
        Apply a set of variable modifications on an array of peptides.

        :param seq_indices: (ndarray int64) array of base sequence indices
        :param mod_arrs: (ndarray uint8) input modification arrays
        :param site_info_idx: (ndarray int64) indices into the site info table
        :param lin_arr: (ndarray bool) linear only peptide
        :param site_info_table: (struct ndarray) site info table
            sequence_index (intp) index into the base sequence
            sites_first_idx (intp) first index into peptide sites (Searcher.sites)
            sites_last_idx (intp) last index into peptide sites (Searcher.sites)
            nterm (bool_) if the peptide occurs n-terminal (on any protein)
            cterm (bool_) if the peptide occurs c-terminal (on any protein)
            nterm_aa_block (bool_) nterm aa is blocked on all proteins
            cterm_aa_block (bool_) cterm aa is blocked on all proteins
        :param modifications: (list Modification) list of modifications to apply
        :param max_mods: (int) Maximum number of modifications to apply on each peptide
        :param level: (str) 'protein' or 'peptide' level modifications
        :param linear_only: (bool) resulting peptides will be flagged as linear_only

        return: sequence indices, modification, sites and linear only arrays of modified peptides
        rtype: ndarray int64, ndarray uint8, struct ndarray, ndarray bool
        """
        # find the matches between peptides and modifications
        matches = self.create_matches_array(
            seq_indices, mod_arrs, site_info_table[site_info_idx], modifications, level)

        # loop over number of modifications per peptide
        return_seq_idx = [seq_indices]
        return_mod_arrs = [mod_arrs]
        return_site_info_idx = [site_info_idx]
        return_lin_arrs = [lin_arr]

        matches = [matches]
        mod_idx_arr = [m.index for m in modifications]
        for n in range(max_mods):
            n_seq_idx, n_mod_arrs, n_site_info_idx, n_match, n_lin_arrs = \
                self.apply_modifications_n(matches[n], return_seq_idx[n], return_mod_arrs[n],
                                           return_site_info_idx[n], return_lin_arrs[n],
                                           mod_idx_arr)
            if len(n_seq_idx) > 0:
                # peptides that have reached max mod peps should no longer be modified
                n_match[self.mod_pep_count[n_seq_idx] >= self.config.max_modified_peps] = False
                return_seq_idx.append(n_seq_idx)
                return_mod_arrs.append(n_mod_arrs)
                return_site_info_idx.append(n_site_info_idx)
                if linear_only:
                    return_lin_arrs.append(np.full(len(n_seq_idx), True))
                else:
                    return_lin_arrs.append(n_lin_arrs)
                matches.append(n_match)
            else:
                break

        return_seq_idx = np.hstack(return_seq_idx)
        return_mod_arrs = np.vstack(return_mod_arrs)
        return_site_info_idx = np.hstack(return_site_info_idx)
        return_lin_arrs = np.hstack(return_lin_arrs)

        return return_seq_idx, return_mod_arrs, return_site_info_idx, return_lin_arrs

    def apply_modifications_n(self, matches, seq_idx_arr, mod_arr, site_info_idx_arr, lin_arr,
                              mod_idx_arr):
        """
        Apply modifications on a peptide.

        :param matches: (ndarray, ndim=3) match array
            dim 1: peptides
            dim 2: amino acids
            dim 3: modifications
        :param seq_idx_arr: (ndarray int64) array of base sequence indices
        :param mod_arr: (ndarray uint8) input modification arrays
        :param site_info_idx_arr: (ndarray int64) indices into the site info table
        :param lin_arr: (ndarray bool) linear only flag of input peptides
        :param mod_idx_arr: (ndarray) indices of the modifications to apply
        return: sequence indices, modification, site_info_idx and linear only arrays of mod peptides
        rtype: ndarray int64, ndarray uint8, ndarray int64, ndarray bool
        """
        return_mod_arrs = []
        return_seq_idx = []
        return_site_info_idx = []
        return_matches = []
        return_lin_arr = []

        # loop over modification dimension (2)
        for i in range(matches.shape[2]):
            mod_matches = matches[:, :, i]
            mod_index = mod_idx_arr[i]
            # loop over amino acid position dimension (1)
            for aa_pos in range(mod_matches.shape[1]):
                new_mod_mask = mod_matches[:, aa_pos]
                if any(new_mod_mask):
                    new_seq_idx = seq_idx_arr[new_mod_mask]
                    new_site_info_idx = site_info_idx_arr[new_mod_mask]
                    new_lin = lin_arr[new_mod_mask]
                    new_match = np.copy(matches[new_mod_mask])
                    new_mod_arrs = np.copy(mod_arr[new_mod_mask])

                    # set the modification index on this amino acid position
                    new_mod_arrs[:, aa_pos] = mod_index
                    # set positions up to aa_pos to False to prevent duplicates
                    new_match[:, :aa_pos + 1] = False

                    # sort the arrays so we can use unique and check mod count per seq_idx
                    sort_order = np.argsort(new_seq_idx)
                    new_seq_idx = new_seq_idx[sort_order]
                    new_site_info_idx = new_site_info_idx[sort_order]
                    new_lin = new_lin[sort_order]
                    new_mod_arrs = new_mod_arrs[sort_order]
                    new_match = new_match[sort_order]

                    # check how many mod peptides are still allowed for each seq_idx
                    seq_idx, rvs, count = np.unique(new_seq_idx, return_inverse=True,
                                                    return_counts=True)
                    # create a mask to only append up to max mod new peptides per seq_idx
                    concat_count = concatenated_ranges(count)
                    max_mask = self.config.max_modified_peps - self.mod_pep_count[
                        seq_idx[rvs]] > concat_count

                    # increase the mod peptide counts for these base sequence_indices
                    # Note: count is increased above max_modified_peps could get the actual appended
                    # count (np.unique(rvs[max_mask], return_counts=True), but we don't really care
                    # about it.
                    self.mod_pep_count[seq_idx] += count

                    # set match mask to False for all peptides that have reached max count
                    new_match[(self.mod_pep_count >= self.config.max_modified_peps)[rvs]] = False
                    new_match[~max_mask] = False

                    return_seq_idx.extend(new_seq_idx[max_mask])
                    return_mod_arrs.extend(new_mod_arrs[max_mask])
                    return_site_info_idx.extend(new_site_info_idx[max_mask])
                    return_matches.extend(new_match[max_mask])
                    return_lin_arr.extend(new_lin[max_mask])

        if len(return_seq_idx) > 0:
            return_seq_idx = np.array(return_seq_idx)
            return_mod_arrs = np.vstack(return_mod_arrs)
            return_site_info_idx = np.hstack(return_site_info_idx)
            return_matches = np.array(return_matches)
            return_lin_arr = np.hstack(return_lin_arr)

        return return_seq_idx, return_mod_arrs, return_site_info_idx, return_matches, return_lin_arr


def modified_sequence_strings(sequences, modified_peptides, config, mod_position='modX'):
    """
    Generate modified sequence strings from array form.

    :param sequences: (ndarray, bytes) unique unmodified sequences
    :param modified_peptides: modified peptide array
    :param config: search configuration
    :param mod_position: Syntax to use for writing modified peptide sequences out
    """
    mod_names = [b''] + [mod.name.encode('ascii')
                         for mod in config.modification.modifications]

    modifications = modified_peptides['modifications']
    sequence_indices = modified_peptides['sequence_index']

    if mod_position == 'modX':
        return np.array(
            [b''.join(
                # n-terminal modification
                [mod_names[modifications[i, 0]]]
                # modified AAs
                + [b''.join([mod_names[mod], bytes([aa])])
                   for aa, mod in zip(sequences[sequence_indices[i]], modifications[i, 2:])]
                # c-terminal modification
                + [mod_names[modifications[i, 1]]])
             for i in range(len(sequence_indices))]
        )
    # Xmod
    else:
        return np.array(
            [b''.join(
                # n-terminal modification
                [mod_names[modifications[i, 0]]]
                # modified AAs
                + [b''.join([bytes([aa]), mod_names[mod]])
                   for aa, mod in zip(sequences[sequence_indices[i]], modifications[i, 2:])]
                # c-terminal modification
                + [mod_names[modifications[i, 1]]])
             for i in range(len(sequence_indices))]
        )


def modified_peptides_from_fixed_mod_sequences(base_sequences, fixed_mod_sequences, config):
    """
    Generate array form of modified peptides for fixed (or known) modified sequence strings.

    :param base_sequences: (ndarray, bytes) original unmodified sequences
    :param fixed_mod_sequences: (ndarray, str) sequences modified with protein fixed modifications
    :param config: (Config) search configuration

    :return: Table of the format (peptide_index, [nterm, cterm, AA0, .., AA-1])
             The values for nterm, cterm and AA refer to specific modifications (int).
    :rtype: np.ndarray
    """
    mod_indices = {mod.name: i+1 for i, mod in enumerate(config.modification.modifications) if
                   (mod.type == 'fixed' or mod.type == 'known') and mod.level == 'protein'}
    mod_names = list(mod_indices.keys())

    if len(set(mod_names)) != len(mod_names):
        raise Exception('Duplicate names in fixed/known protein level modifications detected!')

    mod_indices['H-'] = 0
    mod_indices['-OH'] = 0
    mod_indices[''] = 0

    result_dtype = [('sequence_index', np.intp),
                    ('modifications', np.uint8, (base_sequences.dtype.itemsize + 2,))]

    modified_peptides = np.zeros(len(fixed_mod_sequences), result_dtype)

    modifications = modified_peptides['modifications']
    sequence_indices = modified_peptides['sequence_index']

    parser_labels = parser.std_labels + ['U', 'O', 'X'] + mod_names

    bar = ProgressBar("Converting modified sequences from modX to array format",
                      len(fixed_mod_sequences))
    for i, mod_sequence in enumerate(fixed_mod_sequences):
        parts = parser.parse(mod_sequence.decode('ascii'), show_unmodified_termini=True,
                             labels=parser_labels)
        nterm = parts[0]
        cterm = parts[-1]
        AAs = parts[1:-1]
        modifications[i, 0] = mod_indices[nterm]
        modifications[i, 1] = mod_indices[cterm]
        modifications[i, 2:len(parts)] = [mod_indices[AA[:-1]] for AA in AAs]
        sequence = ''.join([AA[-1] for AA in AAs]).encode('ascii')
        sequence_indices[i] = np.searchsorted(base_sequences, sequence)
        bar.next()
    bar.finish()

    return modified_peptides
