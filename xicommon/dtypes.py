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

# todo - may need split into common and search specific
"""Central place for reused numpy data types."""
import numpy as np


fragment_sites = np.dtype([
    ('peptide_index', np.intp),     # index into peptide sequences array
    ('fragment_index', np.intp),    # index into unique fragment sequences array
    ('nterm', np.bool_),            # n-terminal fragment
    ('cterm', np.bool_),            # c-terminal fragment
    ('start', np.intp),             # start index into peptide amino acids
    ('end', np.intp)                # end index into peptide amino acids
])

peptide_sites = np.dtype([
    ('peptide_id', np.intp),        # index into peptide sequences array
    ('protein_id', np.intp),        # index into protein array
    ('start', np.intp),             # start amino acid position in the protein
    ('end', np.intp),               # end amino acid position in the protein
    ('nterm', np.bool_),            # whether this site is at the n-terminus of the protein
    ('cterm', np.bool_),            # whether this site is at the c-terminus of the protein
    ('nterm_aa_block', np.bool_),   # whether the nterm aa is crucial to digestion (no link or mod)
    ('cterm_aa_block', np.bool_)    # whether the cterm aa is crucial to digestion (no link or mod)
])

basic_fragments = np.dtype([
    ('mz', np.float64),             # fragment m/z value
    ('ion_type', '<S1'),            # ion type of the fragment, e.g. 'b', 'y' or 'P'
    ('term', '<S1'),                # terminus of the fragment ('n' or 'c')
    ('idx', np.uint8),              # offset into peptide from the terminus
    ('start', np.uint8),            # start index in the peptide
    ('end', np.uint8),              # end index in the peptide
])

peak_cluster = np.dtype([
    ('cluster_id', np.uint16),      # id of the cluster
    ('peak_id', np.uint16)          # a peak that belongs to the cluster
])

fragments = np.dtype([
    ('mz', np.float64),             # fragment m/z value
    ('charge', np.uint8),           # fragment charge state
    ('LN', np.bool_),               # linear (True) or crosslinked fragment (False)
    ('loss', '<S25'),               # name of the neutral loss or b'' for primary fragment
    ('nlosses', np.uint8),          # number of times the loss occurred
    ('stub', '<S1'),                # cleaved crosslinker stub type
    ('ion_type', '<S1'),            # ion type of the fragment, e.g. b'b', b'y' or b'P'
    ('term', '<S1'),                # terminus of the fragment (b'n' or b'c')
    ('idx', np.uint8),              # fragment number
    ('pep_id', np.uint8),           # which peptide this fragment belongs to
    ('ranges', np.uint8, (2, 2))    # range over the two peptides as 2x2 array (start, end)
])

annotations = np.dtype([
    ('cluster_id', np.uint16),               # which cluster is matched
    ('cluster_charge', np.int8),            # what charge had the cluster assigned
    ('frag_mz', fragments['mz']),           # fragment m/z value
    ('frag_charge', fragments['charge']),   # fragment charge state
    ('LN', fragments['LN']),                # linear (True) or crosslinked fragment (False)
    ('loss', fragments['loss']),            # name of the neutral loss or '' for primary fragment
    ('nlosses', fragments['nlosses']),      # number of times the loss occurred
    ('stub', fragments['stub']),            # cleaved crosslinker stub type
    ('ion_type', fragments['ion_type']),    # ion type of the fragment, e.g. 'b', 'y' or 'P'
    ('term', fragments['term']),            # terminus of the fragment ('n', 'c' or 'P')
    ('idx', fragments['idx']),              # fragment number
    ('pep_id', fragments['pep_id']),        # which peptide this fragment belongs to (1/2)
    ('ranges', fragments['ranges']),        # range over the peptides as 2x2 array (start, end)
    ('peak_mz', np.float64),                # matched peak m/z
    ('peak_int', np.float64),               # matched peak int
    ('rel_error', np.float64),              # relative match error
    ('abs_error', np.float64),              # absolute match error
    ('missing_monoisotopic_peak', bool),    # was the peak matched against the first isotopic peak
    ('alpha_beta_index', np.intp)           # id of the alpha_beta_candidate
])

# used for the protein link sites
prot_link_sites = np.dtype([
    ('alpha_beta_index', np.intp),  # id of the alpha_beta_candidate
    ('prot1', np.intp),  # id of protein1
    ('prot2', np.intp),  # id of protein2
    ('pep_pos1', np.intp),  # position of the peptide in protein1
    ('pep_pos2', np.intp),  # position of the peptide in protein2
    ('link_pos1', np.intp),  # position of the linked AA in protein1
    ('link_pos2', np.intp),  # position of the linked AA in protein2
])

# return dtype of link_site_annotations()
link_site_annotations = np.dtype([
    ('alpha_beta_index', np.intp),  # which alpha-beta candidate match is referenced
    ('link1', np.int16),    # AA position of crosslinked residue on the alpha peptide
    ('link2', np.int16),    # AA position of crosslinked residue on the beta peptide
    ('score', np.float32),  # score for these link sites
    ('annotation', object)   # annotations for these link sites
])

link_score = np.dtype([
    ('alpha_beta_index', np.intp),  # which alpha-beta candidate match is referenced
    ('link_p1', np.int16),  # AA position of crosslinked residue on peptide1
    ('link_p2', np.int16),  # AA position of crosslinked residue on peptide2
    ('score', np.float32)  # score for these link sites
])


beta_candidate = np.dtype([
    ('peptide_index', np.intp),
    ('isotope_index', np.intp),
    ('crosslinker_index', np.intp)
])

ab_candidate = np.dtype([
    ('alpha_pep_index', np.intp),
    ('beta_pep_index', np.intp),
    ('isotope_index', np.intp),
    ('crosslinker_index', np.intp)
])

site_info_table = np.dtype([
    ('sequence_index', np.intp),  # index into the base sequence
    ('sites_first_idx', np.intp),  # first index into peptide sites (Searcher.sites)
    ('sites_last_idx', np.intp),  # last index into peptide sites (Searcher.sites)
    ('nterm', np.bool_),  # if the peptide occurs n-terminal (on any protein)
    ('cterm', np.bool_),  # if the peptide occurs c-terminal (on any protein)
    ('nterm_aa_block', np.bool_),  # nterm aa is blocked on all proteins
    ('cterm_aa_block', np.bool_),  # cterm aa is blocked on all proteins
])

scores_table_dtype = [
    ('linear', bool),  # linear or crosslinked peptide
    ('alpha_id', np.uint8),  # alpha pep_id (1 or 2, if alpha pep is p1 or p2)
    ('alpha_index', np.intp),  # index into the peptide_db for the alpha peptide
    ('alpha_score', np.float32),  # score of the alpha peptide
    ('alpha_rank', np.uint16),  # rank of the alpha peptide
    ('beta_id', np.uint8),  # beta pep_id (1 or 2, if beta pep is p1 or p2)
    ('beta_index', np.intp),  # index into the peptide_db for the beta peptide
    ('beta_score', np.float32),  # score of the beta peptide
    ('beta_count', np.uint16),  # number of beta peptides found for this alpha peptide
    # with the same number of missing precursor isotope peaks (ximpa)
    ('beta_count_inverse', np.float32),  # inverse of beta_count
    ('beta_count_total', np.uint16),  # number of beta peptides found for this alpha peptide
    ('beta_count_inverse_total', np.float32),  # inverse of beta_count_total
    ('alpha_beta_index', np.intp),  # index of the alpha beta candidate
    ('alpha_beta_score', np.float32),  # score of the alpha beta candidate
    ('alpha_beta_rank', np.uint16),  # rank of the alpha beta candidate
    ('p1_index', np.intp),  # index into the peptide_db for p1 (better explained pep)
    ('mass_p1', np.float64),  # neutral mass of p1
    ('linked_aa_p1', '<U1'),  # Amino acid that is linked on p1
    ('link_pos_p1', np.int16),  # linked AA residue position in p1
    ('decoy_p1', bool),  # if p1 is a decoy peptide
    ('p2_index', np.intp),  # index into the peptide_db for p2 (worse explained pep)
    ('mass_p2', np.float64),  # neutral mass of p2
    ('linked_aa_p2', '<U1'),  # Amino acid that is linked on p2
    ('link_pos_p2', np.int16),  # linked AA residue position in p2
    ('decoy_p2', bool),  # if p2 is a decoy peptide
    ('precursor_mz', np.float64),  # m/z of the precursor (potentially corrected)
    ('precursor_missing_isotope_peak', np.uint8),  # number of missing isotopic peaks
    ('precursor_charge', np.int16),  # precursor charge state (potentially corrected)
    ('precursor_mass', np.float64),  # neutral mass of the precursor
    ('crosslinker_index', np.int8),  # index into the configured cross-linker
]
