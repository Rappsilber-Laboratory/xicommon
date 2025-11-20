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

from xicommon.config import Config
from xicommon.context_base import ContextBase
from xicommon.simple_databases import SimplePeptideDatabase, SimpleFragmentDatabase
from xicommon.cython import Fragmentation
from xicommon import dtypes
from xicommon.modifications import Modifier
import re
from numpy.lib import recfunctions
import numpy as np


def modified_peptides_from_sequences(base_sequences, modified_sequences, config):
    """
    Generate array form of modified peptides from original and modified sequences.

    Note this assumes unique modification names!
    :param base_sequences: (ndarray, bytes) original unmodified sequences
    :param modified_sequences: (ndarray, str) modified sequences
    :param config: (Config) search configuration

    :return: Table of the format (peptide_index, [nterm, cterm, AA0, .., AA-1])
             The values for nterm, cterm and AA refer to specific modifications (int).
    :rtype: np.ndarray
    """
    mod_names = [''] + [mod.name for mod in config.modification.modifications]

    if len(set(mod_names)) != len(mod_names):
        raise Exception('Creating the MockContext peptide database assumes unambiguous '
                        'modification names!')

    mod_indices = {name: i for i, name in enumerate(mod_names)}
    mod_indices['H-'] = 0
    mod_indices['-OH'] = 0

    result_dtype = [('sequence_index', np.intp),
                    ('modifications', np.uint8, (base_sequences.dtype.itemsize + 2,))]

    modified_peptides = np.zeros(len(modified_sequences), result_dtype)

    modifications = modified_peptides['modifications']
    sequence_indices = modified_peptides['sequence_index']
    # split the sequence into parts - currently hardcoded for modX format
    seqpat = re.compile(r"""(?:
            # optional n-term mod or empty string
            ^(?:[^A-Z]+-)?
             # optional c-term mod or empty string  '(?<=[A-Z]))' prevents double matching $
            |(?:-[^A-Z]+|(?<=[A-Z]))$
            # any aminoacid with potential modification
            |(?:[^A-Z]*[A-Z])
            )
            """, re.X)
    for i, mod_sequence in enumerate(modified_sequences):
        # split the mod_sequence into nterm, modified AAs, and cterm
        parts = re.findall(seqpat, mod_sequence.decode('ascii'))

        nterm = parts[0]
        cterm = parts[-1]
        AAs = parts[1:-1]
        modifications[i, 0] = mod_indices[nterm]
        modifications[i, 1] = mod_indices[cterm]
        modifications[i, 2:len(parts)] = [mod_indices[AA[:-1]] for AA in AAs]
        sequence = ''.join([AA[-1] for AA in AAs]).encode('ascii')
        sequence_indices[i] = np.searchsorted(base_sequences, sequence)

    return modified_peptides


class MockContext(ContextBase):
    def __init__(self, config=Config(), all_tasks=1, this_task=0):
        super().__init__(config)
        self.peptide_db = None
        self.fragment_db = None
        self.peptide_aa_lengths = None
        self.modified_peptides_aa_lengths = None
        self.modified_peptides = None
        self.unmodified_peptide_sequences = None
        self.site_info_table = None
        self.fixed_mod_peptide_sequences = None
        self.modifier = Modifier(self)
        self.fragmentation = Fragmentation(self)
        self.cluster = {"all_tasks": all_tasks, "this_task": this_task}

    def setup_peptide_db_xi2annotator(self, base_sequences, modification_ids,
                                      modification_positions):
        """
        Setup the peptide database for the xi2annotator .

        Based on base sequences, modification ids and positions.
        """
        # make unique and sort alphabetically
        self.unmodified_peptide_sequences = np.unique(base_sequences)

        # create modified peptides from the peptide sequences, modification ids and positions
        result_dtype = [('sequence_index', np.intp),
                        ('modifications', np.uint8,
                         (self.unmodified_peptide_sequences.dtype.itemsize + 2,))]
        self.modified_peptides = np.zeros(len(base_sequences), result_dtype)

        for i, (mod_ids, mod_positions, base_seq) in enumerate(
                zip(modification_ids, modification_positions, base_sequences)):
            self.modified_peptides['sequence_index'][i] = np.searchsorted(
                self.unmodified_peptide_sequences, base_seq)
            # input modification positions are 0-based, -1 is n-terminal, 32767 is c-terminal
            # xi2 modified peptides format (0 is n-terminal, 1 is c-terminal, then AAs)
            #
            # input modification ids: 0 is first modification in config, 1 is second, etc.
            # xi2 modification ids: 0 is unmodified, 1 is first modification in config, etc.
            # so we need to add 1 to the input modification ids
            for mod_id, mod_position in zip(mod_ids, mod_positions):
                if mod_position == 0:
                    self.modified_peptides['modifications'][i, 0] = mod_id + 1
                elif mod_position == 32767:
                    self.modified_peptides['modifications'][i, 1] = mod_id + 1
                else:
                    self.modified_peptides['modifications'][i, mod_position + 1] = mod_id + 1

        # sort them, so they match the unmodified_peptide_sequences array
        self.modified_peptides.sort()

        self.modified_peptides = recfunctions.append_fields(
            self.modified_peptides, names=['linear_only', 'var_mod_count'],
            dtypes=[np.bool_, np.uint8],
            data=[np.zeros(self.modified_peptides.size)] * 2, usemask=False)

        self.peptide_db = SimplePeptideDatabase(self)
        self.peptide_aa_lengths = np.char.str_len(self.unmodified_peptide_sequences)
        self.modified_peptides_aa_lengths = self.peptide_aa_lengths[
            self.modified_peptides['sequence_index']]

    def setup_peptide_db(self, peptides, as_nterminal=False, as_cterminal=False,
                         modification_ids=None, modification_positions=None):
        self.fixed_mod_peptide_sequences = peptides

        # make unique and sort alphabetically
        self.unmodified_peptide_sequences = np.unique(
            [re.sub(b'[^A-Z]', b'', sequence) for sequence in peptides])

        if modification_ids is None or modification_positions is None:
            self.modified_peptides = modified_peptides_from_sequences(
                self.unmodified_peptide_sequences, peptides, self.config)
        else:
            # create modified peptides from the peptide sequences, modification ids and positions
            result_dtype = [('sequence_index', np.intp),
                            ('modifications', np.uint8,
                             (self.unmodified_peptide_sequences.dtype.itemsize + 2,))]
            self.modified_peptides = np.zeros(len(peptides), result_dtype)

            # input modification positions are 1-based, 0 is n-terminal, len+1 is c-terminal
            # convert to xi2 modified peptides format (0 is n-terminal, 1 is c-terminal, then AAs)
            for i, (mod_ids, mod_positions, pep_seq) in enumerate(
                    zip(modification_ids, modification_positions, peptides)):
                pep_seq = re.sub(b'[^A-Z]', b'', pep_seq)
                self.modified_peptides['sequence_index'][i] = np.searchsorted(
                    self.unmodified_peptide_sequences, pep_seq)
                for mod_id, mod_position in zip(mod_ids, mod_positions):
                    if mod_position == 0:
                        self.modified_peptides['modifications'][i, 0] = mod_id + 1
                    elif mod_position == len(peptides[i]) + 1:
                        self.modified_peptides['modifications'][i, 1] = mod_id + 1
                    else:
                        self.modified_peptides['modifications'][i, mod_position + 1] = mod_id + 1

        # sort them, so they match the unmodified_peptide_sequences array
        self.modified_peptides.sort()

        self.modified_peptides = recfunctions.append_fields(
            self.modified_peptides, names=['linear_only', 'var_mod_count'],
            dtypes=[np.bool_, np.uint8],
            data=[np.zeros(self.modified_peptides.size)] * 2, usemask=False)

        # calculate var_mod_count
        var_mod_idxs = np.array([m.index for m in self.config.modification.modifications if
                                 m.type == 'variable'])
        self.modified_peptides['var_mod_count'] = np.any(
            self.modified_peptides['modifications'] == var_mod_idxs.reshape(-1, 1, 1),
            axis=0).sum(axis=1)

        self.peptide_db = SimplePeptideDatabase(self)
        self.peptide_aa_lengths = np.char.str_len(self.unmodified_peptide_sequences)
        self.modified_peptides_aa_lengths = self.peptide_aa_lengths[
            self.modified_peptides['sequence_index']]
        # Build modified peptide array
        self.site_info_table = np.empty(len(peptides), dtype=dtypes.site_info_table)
        self.site_info_table['sequence_index'] = np.arange(peptides.size)
        self.site_info_table['sites_first_idx'] = np.arange(peptides.size)
        self.site_info_table['sites_last_idx'] = np.arange(peptides.size)
        self.site_info_table['nterm'] = as_nterminal
        self.site_info_table['cterm'] = as_cterminal
        self.site_info_table['nterm_aa_block'] = False
        self.site_info_table['cterm_aa_block'] = False

        self.setup_fragment_db(self.unmodified_peptide_sequences)

    def setup_fragment_db(self, unmod_sequences):
        if self.modified_peptides is None:
            self.modified_peptides = np.empty(len(unmod_sequences), dtype=[
                ('sequence_index', np.intp),
                ('modifications', np.uint8, (unmod_sequences.dtype.itemsize + 2,))])
            self.modified_peptides['sequence_index'] = np.arange(len(unmod_sequences))
            self.modified_peptides['modifications'] = 0
        self.fragment_db = SimpleFragmentDatabase(self)

    def get_site_info_for_mod_pep_index(self, pep_index):
        return self.site_info_table[pep_index]
