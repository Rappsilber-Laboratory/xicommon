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

"""
Central module for the fragmentation of peptides.

This module generates specific fragments for ions, losses, modified
fragments and unmodified sequences.
"""
from xicommon.utils import concatenated_ranges, get_chunk_indices
from xicommon.mass import mass, ion_mass_byte, unmodified_termini_mass
from xicommon import dtypes
from xicommon.const import PROTON_MASS
from xicommon.xi_logging import log
from xicommon.cython import fast_unique, unique_byte_ranges, isin_set_schar
import numpy as np


nterm_or_peptide = np.array([b'n', b'P'])
cterm_or_peptide = np.array([b'c', b'P'])


def fragment_sequences(peptide_sequences, add_precursor=False):
    """
    Generate fragments for unmodified peptide sequences.

    :param peptide_sequences : (ndarray, bytes) peptide sequences
    :param add_precursor : (bool) Flag indicating if precursor sequences should be added
    :return:
        fragments: (ndarray, bytes) unique fragment sequences
        sites: (ndarray) sites each fragment was found, fields as follows:
            peptide_index (int) index into peptide sequences
            fragment_index (int) index into unique fragment sequences
            nterm (bool) n-terminal fragment
            cterm (bool) c-terminal fragment
            start (int) start index into peptide amino acids
            end (int) end index into peptide amino acids
    """
    # n peptides
    npeptides = peptide_sequences.size

    log("Generating fragment sequences from %d peptide sequences" % npeptides)

    # Peptide sequences and lengths (array)
    peptide_lengths = np.char.str_len(peptide_sequences)

    # Number of fragment sites per terminal in each peptide
    fragments_per_terminal = peptide_lengths - 1

    # Number of fragmentation sites for each peptide
    # sites table size depends on number of fragments and if precursors should be added
    term_site_counts = fragments_per_terminal * 2
    if add_precursor:
        term_site_counts += 1

    # Allocate table for fragment sites
    sites = np.empty(term_site_counts.sum(), dtype=dtypes.fragment_sites)

    log("Identified %d possible fragmentation sites" % len(sites))

    # Generate site spans and subtables for easy access to the respective rows for n/c-term sites
    # for the site assignment we only need the n/c-term sites not the precursor idx / cells
    # these are the non-precursor rows or n/c-term rows
    if add_precursor:
        # these are the precursor rows
        prec_sites = sites[-npeptides:]

        # set precursor ids
        prec_sites["peptide_index"] = np.arange(npeptides)

        # define precursors as neither n-terminal nor c-terminal
        # fixme: redefine precursor as n and c-terminal True?
        prec_sites['nterm'] = False
        prec_sites['cterm'] = False

        # add precursor data, its always the complete peptide sequence
        prec_sites["start"] = 0
        prec_sites["end"] = peptide_lengths

        # term_sites holds the n and c-terminal sites
        term_sites = sites[:-npeptides]

        prec_sites["peptide_index"] = np.arange(npeptides)
    else:
        # n-term  & c-term rows
        term_sites = sites

    # assign subtable with n and c-terminal sites
    nterm_sites = term_sites[0::2]
    cterm_sites = term_sites[1::2]

    # Populate peptide indices for the subtable with the n-/c- terminal sequences
    term_sites["peptide_index"] = np.repeat(np.arange(npeptides), fragments_per_terminal * 2)

    ranges = concatenated_ranges(peptide_lengths - 1) + 1
    nterm_sites['start'] = 0
    nterm_sites['end'] = ranges
    cterm_sites['start'] = ranges
    cterm_sites['end'] = np.repeat(peptide_lengths, fragments_per_terminal)

    # n-terminal sequences start always at position 0
    # c-terminal sequences are the opposite
    nterm_sites['nterm'] = True
    nterm_sites['cterm'] = False
    cterm_sites['nterm'] = False
    cterm_sites['cterm'] = True

    # Reinterpret peptide sequences as char array
    peptide_sequence_chars = peptide_sequences.view(np.uint8).reshape(npeptides, -1)

    # Find unique fragment sequences
    log("Finding unique fragment sequences")
    unique_fragment_chars, fragment_indices = \
        unique_byte_ranges(peptide_sequence_chars,
                           sites['peptide_index'],
                           sites['start'], sites['end'])

    log("Found %d unique fragment sequences" % len(unique_fragment_chars))

    # Recast as string array
    fragments = unique_fragment_chars.reshape(-1).view(peptide_sequences.dtype)

    # Populate sequence indices
    log("Populating fragment sequence indices")
    sites['fragment_index'] = fragment_indices
    log("Sorting fragment sites")
    sites.sort()
    return fragments, sites


def modify_fragments(frag_sites, modified_peptides):
    """
    Generate table of modified fragments.

    :param frag_sites: fragment sites
    :param modified_peptides: modified peptide array
    :return:
        (ndarray, int, ndim=1)
            indices of base sequence of the modified fragments
        (ndarray, uint8, ndim=2)
            Modification locations. Shape is N x (L + 2), where N is the number of variations,
            and L is the maximum sequence length. In the second axis, index 0 is the n-terminus and
            1 is the c-terminus. Indices 2 onwards are the amino acids of the sequence. Values
            are zero to indicate no modification, or one plus an index into the config modification
            list, to indicate a modification at that location.
        (ndarray) a new sites array connecting modified fragments to modified peptides:
            peptide_index (int) index into peptide sequences
            fragment_index (int) index into modified fragment sequences
            nterm (bool) n-terminal (True) or c-terminal (False)
            start (int) start index into peptide amino acids
            end (int) end index into peptide amino acids
    """
    peptide_seqs = modified_peptides['sequence_index']
    peptide_mods = modified_peptides['modifications']

    log("Finding unique unmodified peptide indices for %d modified peptides"
        % len(modified_peptides))

    # Unique sequence indices used in modified versions, and how many modifications of each
    unique_seq_indices, mod_counts = fast_unique(peptide_seqs, presorted=True, return_counts=True)

    log("Found %d unique indices, finding fragment site ranges" % len(unique_seq_indices))

    # Get offsets into frag_sites for the modified peptides from each unmodified sequence
    limits = [unique_seq_indices, unique_seq_indices + 1]
    del unique_seq_indices
    frag_starts, frag_ends = np.searchsorted(frag_sites['peptide_index'], limits)
    del limits

    log("Generating indices into original fragment sites for all modified fragments")

    # Repeat these indices for the modified peptides in each group
    mod_frag_starts, mod_frag_ends = np.repeat([frag_starts, frag_ends], mod_counts, axis=1)

    # No longer need these counts
    del mod_counts

    # Fragment counts for each group of modified peptides
    mod_frag_counts = mod_frag_ends - mod_frag_starts

    # Get indices into original fragmentation sites for each modified fragment
    mod_site_indices = get_chunk_indices(mod_frag_starts, mod_frag_ends, mod_frag_counts)
    num_mod_fragments = len(mod_site_indices)

    # No longer need these limits
    del mod_frag_starts, mod_frag_ends

    log("Found %d total sites of modified fragments" % num_mod_fragments)

    # Create new sites array for modified peptides
    log("Building array of modified fragmentation site data")
    mod_frag_sites = frag_sites[mod_site_indices]

    # No longer need these indices
    del mod_site_indices

    # Refer new sites array to modified peptides
    log("Populating modified peptide indices for modified fragmentation sites")
    mod_frag_sites['peptide_index'] = np.repeat(np.arange(len(peptide_seqs)), mod_frag_counts)

    # No longer need these counts
    del mod_frag_counts

    # Transfer terminal modifications
    # select nterminal sequences and the precursor
    log("Populating fragment terminal modifications")
    term_mods = np.zeros((num_mod_fragments, 2), np.uint8)
    precursor = (mod_frag_sites['nterm'] == mod_frag_sites['cterm'])
    nterm_mask = mod_frag_sites['nterm'] | precursor
    cterm_mask = mod_frag_sites['cterm'] | precursor
    term_mods[nterm_mask, 0] = peptide_mods[mod_frag_sites['peptide_index'][nterm_mask], 0]
    term_mods[cterm_mask, 1] = peptide_mods[mod_frag_sites['peptide_index'][cterm_mask], 1]

    # Get unique AA modification sets across all fragments, and mapping from sites to these.
    log("Finding unique AA modification sets")
    unique_mod_sets, mod_set_indices = unique_byte_ranges(peptide_mods[:, 2:],
                                                          mod_frag_sites['peptide_index'],
                                                          mod_frag_sites['start'],
                                                          mod_frag_sites['end'])
    log("Found %d unique AA modification sets" % len(unique_mod_sets))

    # Build table used to find unique combinations
    log("Building table to find unique modified fragments")
    table = np.empty(num_mod_fragments, dtype=[
        ('fragment_index', mod_frag_sites['fragment_index'].dtype),
        ('nterm', bool),
        ('term_mods', np.uint8, (2,)),
        ('aa_modification_set_index', np.intp)])
    table['fragment_index'] = mod_frag_sites['fragment_index']
    table['nterm'] = mod_frag_sites['nterm']
    table['term_mods'] = term_mods
    table['aa_modification_set_index'] = mod_set_indices

    log("Finding order for table")
    table_bytes = table.view(np.uint8).reshape(num_mod_fragments, -1)
    table_order = np.lexsort(table_bytes.T[::-1])

    log("Finding unique modified fragments from %d table entries" % len(table))
    # Find unique modified fragments, and indexes of these for all modified fragment sites
    unique_entries, modified_to_unique = fast_unique(table, order=table_order, return_inverse=True)
    log("Found %d unique modified fragments" % len(unique_entries))

    del table, table_bytes

    # Sequence indices for unique modified fragments
    unique_frag_sequence_indices = unique_entries['fragment_index']

    # Modifications for unique modified fragments
    unique_frag_mods = np.empty((len(unique_entries), peptide_mods.shape[1]), np.uint8)
    unique_frag_mods[:, :2] = unique_entries['term_mods']
    unique_frag_mods[:, 2:] = unique_mod_sets[unique_entries['aa_modification_set_index']]

    # Refer frag_sites to unique modified fragments
    log("Referencing modified fragment sites to original fragments")
    mod_frag_sites['fragment_index'] = modified_to_unique

    return unique_frag_sequence_indices, unique_frag_mods, mod_frag_sites


def fragment_ions(sequences, modified_peptides, context, add_precursor=False):
    """
    Return a generator that produces the fragment ions.

    Yields once for each ion type.
    :param sequences: (bytes ndarray) array of unmodified peptide sequences
    :param modified_peptides: (ndarray) array describing modified peptides
    :param context: (Searcher) Search context
    :param add_precursor: (bool) If True generates precursor ions
    :return: (generator) returns a generator yielding:
        (str) terminus, e.g. 'n' or 'c',
        (str) ion type, e.g. 'b',
        (Loss or None) neutral loss,
        (int) loss count,
        (struct ndarray): fragment sites
            [('peptide_index', '<i8'), ('fragment_index', '<i8'), ('nterm', '?'), ('start', '<i8'),
             ('end', '<i8')],
        (float64, ndarray): fragment masses
    """
    config = context.config

    # Get fragments of unmodified sequences
    fragments, orig_sites = fragment_sequences(sequences, add_precursor=add_precursor)

    # Find fragment sequence masses
    frag_seq_masses = mass(fragments, config=config)

    # Delete the fragment sequences - we only need the masses from here on.
    del fragments

    # Get modified fragments
    frag_indices, frag_mods, mod_frag_sites = modify_fragments(orig_sites,
                                                               modified_peptides)
    # No longer need the sites of the unmodified fragments
    del orig_sites

    # Get modification masses
    log("Calculating fragment masses")
    frag_mod_masses = mass(sequences=None,
                           sequence_indices=None,
                           modifications=frag_mods,
                           config=config)

    # No longer need the modifications now
    del frag_mods

    # Get total fragment masses
    frag_masses = (frag_seq_masses - unmodified_termini_mass)[frag_indices] + frag_mod_masses

    # Can now delete the separate mass arrays, and the sequence indices
    del frag_seq_masses, frag_indices, frag_mod_masses

    log("Generating fragment ions")
    # Split into nterm, cterm, precursor sites
    # n/c-term columns are True for n- and c-terminal entries
    # if both are false, the entry refers to the precursor
    nterm_mask = mod_frag_sites['nterm']
    cterm_mask = mod_frag_sites['cterm']
    prec_mask = nterm_mask == cterm_mask

    # set the site tables
    nterm_sites = mod_frag_sites[nterm_mask]
    cterm_sites = mod_frag_sites[cterm_mask]
    prec_sites = mod_frag_sites[prec_mask]

    # the following loop needs a list of ions that are usually configured via the
    # config file. To make the loop work with the prec_sites we also add a single
    # entry to the config_prec_ions.
    config_prec_ions = [b"P"] if add_precursor else []

    # Generate ion sites and masses
    for term, term_sites, ions in ((b'n', nterm_sites, config.fragmentation.nterm_ions_ascii),
                                   (b'c', cterm_sites, config.fragmentation.cterm_ions_ascii),
                                   (b'P', prec_sites, config_prec_ions)):
        for ion in ions:
            # b1 ion generation enabled for now - ToDo: Xi-265
            # if ion == 'b':
            #     # No b1 ion possible
            #     valid_sites = term_sites[term_sites['end'] != 1]
            # else:
            #     valid_sites = term_sites
            #
            # # continue if there are no valid sites
            # if valid_sites.size == 0:
            #     continue
            valid_sites = term_sites

            masses = frag_masses + ion_mass_byte(ion)

            frag_ion_masses = masses[valid_sites['fragment_index']]

            yield term, ion, None, 0, valid_sites, frag_ion_masses


def include_losses(fragment_table, peptide_ids, context):
    """
    Return a new array with original fragments and all applicable losses.

    :param fragment_table: (ndarray) primary fragments without losses
    :param peptide_ids: (list) the peptides that these fragments are coming from
    :param context: (Searcher) Search context
    :return:
    """
    for i, loss in enumerate(context.config.fragmentation.losses):
        # expand each list with all losses
        losses = create_loss_fragments(fragment_table, peptide_ids, loss, context)
        fragment_table = np.hstack((fragment_table, losses))

    fragment_table.sort(order=['mz'])
    return fragment_table


def create_loss_fragments(fragment_table, peptide_ids, loss, context):
    """
    Generate a list of neutral loss fragments based on the supplied fragments.

    @TODO needs to be adapted for double fragmentation

    :param fragment_table: (ndarray) primary fragments without losses
    :param peptide_ids: (list) the peptides that these fragments are coming from
    :param loss: (Loss) the type of loss that should be generated
    :param context: (Searcher) Search context
    :return: (ndarray) fragment_table with all the losses of the given type
    """
    # get for each peptide the sites that can lose (disregarding terminals)
    pep_loss_counts = [count_sites_nonterminal(p_id, loss, context) for p_id in peptide_ids]

    # we have to count the number of possible losses for each fragment
    # so initialise with zero losses for each fragment
    loss_counts = np.zeros(fragment_table.size, dtype=np.int64)

    # if we have a terminal loss configured count that for each terminal fragment as 1
    if loss.nterm:
        loss_counts[isin_set_schar(fragment_table['term'].view(np.int8), nterm_or_peptide.view(
            np.int8))] = 1
        # for now cross-linked fragments always contain the whole second peptide -> also a terminal
        loss_counts[~fragment_table['LN']] += 1

    if loss.cterm:
        loss_counts[isin_set_schar(fragment_table['term'].view(np.int8), cterm_or_peptide.view(
            np.int8))] += 1
        # for now cross-linked fragments always contain the whole second peptide -> also a terminal
        loss_counts[~fragment_table['LN']] += 1

    # if we have fragments with losses already - reduce the maximal number of losses for that
    # fragment
    max_losses = context.config.fragmentation.max_nloss - fragment_table['nlosses']

    for i in range(len(peptide_ids)):
        # Prepare an array where each index contains the sum of pep_loss_counts[i]
        # up to but not including that index. So the first entry is zero, and the
        # remaining entries are the cumulative sum of pep_loss_counts[i].
        cumulative_counts = np.empty(len(pep_loss_counts[i]) + 1, dtype=pep_loss_counts[i].dtype)
        cumulative_counts[0] = 0
        cumulative_counts[1:] = np.cumsum(pep_loss_counts[i])
        # Get the starts and ends of the ranges we need to sum over from the fragment table.
        starts, ends = fragment_table['ranges'][:, i].T
        # Get the counts over each range by subtracting the cumulative sum at the start of the
        # range from the cumulative sum at the end of the range.
        pep_total_counts = cumulative_counts[ends] - cumulative_counts[starts]
        # Add the counts for this peptide to our running total.
        loss_counts += pep_total_counts

    loss_counts = np.clip(loss_counts, 0, max_losses)

    lossy_fragments = []

    for count in range(1, loss_counts.max() + 1):
        lname = loss.name + 'x' + str(count)
        lname = lname.encode('ascii')
        # Modified fragments for which this many losses of this type is possible
        loss_count_mask = (loss_counts >= count)

        newlosses = fragment_table[loss_count_mask].copy()
        # we expand some loss-names
        newlosses['loss'][newlosses['nlosses'] > 0] = \
            np.core.defchararray.add(newlosses['loss'][newlosses['nlosses'] > 0], b'_' + lname)
        newlosses['loss'][newlosses['nlosses'] == 0] = lname
        newlosses['nlosses'] += count
        newlosses['mz'] -= loss.mass * count / newlosses['charge']

        lossy_fragments.append(newlosses)

    if len(lossy_fragments) > 0:
        return np.hstack(lossy_fragments)

    return np.array([], dtype=fragment_table.dtype)


def count_sites_nonterminal(pep_index, loss, context):
    """
    Put a 1 for each site that could produce the loss and 0 for sites that could not.

    :param pep_index: (int) Index into the peptide DB for peptide1
    :param loss: (Loss) the loss to check
    :param context (Searcher) Search context
    :return: (ndarray) list containing 1 for loss-producing residues
    """
    aa_spec = loss.ord_aa_specificity
    mod_codes = loss.mod_specificity_codes

    pep_aa_arr = np.array(list(context.peptide_db.unmod_pep_sequence(pep_index)))

    mod_pep = context.peptide_db.peptides[pep_index]
    pep_mod_arr = mod_pep['modifications'][2:pep_aa_arr.size+2]

    aa_spec = np.array(aa_spec).reshape(-1, 1)
    mod_spec = np.array(mod_codes).reshape(-1, 1)
    # check for X (wildcard) in specificity array
    if 88 in aa_spec:
        spec_sites = np.repeat(1, pep_aa_arr.size)
    else:
        # match amino acids for pep1 with linker specificity 1
        aa_spec_matches = pep_aa_arr == aa_spec
        # match modifications for pep1 with linker specificity 1
        mod_spec_matches = pep_mod_arr == mod_spec
        # combine to get possible matches for pep1 with linker specificity 1
        spec_sites = np.any(aa_spec_matches & mod_spec_matches, 0)
        spec_sites.dtype = np.byte

    return spec_sites


def spread_charges(fragment_table, context, max_charge, max_linear_charge=-1,
                   min_crosslinked_charge=0):
    """
    Add all fragments with different charge states.

    Assumes that the input contains all fragments at charge z=1 and then calculates the missing
    charge states for each fragment.
    :param fragment_table: (ndarray) numpy array of all singly charged fragments
    :param max_charge: (int) the maximal charge for fragments
    :param context (Searcher) Search context
    :param max_linear_charge (int) what is the maximum charge state considered for linear fragments
    :param min_crosslinked_charge (int) what is the minimum charge state considered for crosslinked
            fragments
    :return: (ndarray) combined list of all fragments in all charge states
    """
    n_frags = fragment_table.size
    fragments = np.tile(fragment_table, max_charge)
    fragments['charge'] = np.repeat(np.arange(1, max_charge+1), n_frags)
    fragments['mz'] = (fragments['mz'] + (fragments['charge'] - 1)
                       * PROTON_MASS) / fragments['charge']

    if max_linear_charge > -1:
        mask = (~fragments['LN']) | (fragments['charge'] <= max_linear_charge)
        if min_crosslinked_charge > 0:
            mask &= (fragments['LN']) | (fragments['charge'] >= min_crosslinked_charge)
        fragments = fragments[mask]
    elif min_crosslinked_charge > 0:
        mask = (fragments['LN']) | (fragments['charge'] >= min_crosslinked_charge)
        fragments = fragments[mask]

    fragments.sort()

    return fragments
