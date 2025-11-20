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
Central module for generating the fragment m/zs for linear and cross-linked peptides.

This is the central place where sequence specific loss and charge state fragments are generated
from the base fragment db.
"""

from xicommon.const import PROTON_MASS
import numpy as np
from collections.abc import Iterable


def fragment_crosslinked_peptide_pair(pep1_index, pep2_index, link_pos1, link_pos2,
                                      crosslinker, context, add_precursor=False):
    """
    Generate m/z fragments for a crosslinked peptide pair.

    :param pep1_index: (int) Index into the peptide DB for peptide1
    :param pep2_index: (int) Index into the peptide DB for peptide2
    :param link_pos1: (int) 0-based position of the cross-linking site on peptide 1
    :param link_pos2: (int) 0-based position of the cross-linking site on peptide 2
    :param crosslinker: (Crosslinker) the crosslinker linking the peptides
    :param context (Searcher) search context
    :param add_precursor (bool) Adds the precursor fragments if True
    :return: (ndarray) fragments numpy array with mz, charge and ion_type
    """
    # We generate two sets of fragments for the pair:
    # - the possible fragments with a partial pep1 crosslinked to a complete pep2.
    # - the possible fragments with a partial pep2 crosslinked to a complete pep1.
    # If add_precursor is set, we also want to generate the precursor case where
    # pep1 and pep2 are both complete. We don't want to duplicate this case twice,
    # so we arbitrarily choose here to add it for the first set of fragments.
    add_1st_precursor = add_precursor
    add_2nd_precursor = False

    # generate and sort fragments
    pep1_frags = fragment_crosslinked_peptide(pep1_index, pep2_index, link_pos1,
                                              crosslinker, context, add_precursor=add_1st_precursor)
    pep2_frags = fragment_crosslinked_peptide(pep2_index, pep1_index, link_pos2,
                                              crosslinker, context, add_precursor=add_2nd_precursor)
    # set pep_id to 2 for fragments from peptide 2
    pep2_frags['pep_id'] = 2

    # swap the order of ranges for pep2
    pep2_range0 = pep2_frags['ranges'][:, 0].copy()
    pep2_range1 = pep2_frags['ranges'][:, 1].copy()
    pep2_frags['ranges'][:, 0] = pep2_range1
    pep2_frags['ranges'][:, 1] = pep2_range0

    all_frags = np.concatenate([pep1_frags, pep2_frags])

    all_frags.sort()

    return all_frags


def fragment_noncovalent_peptide_pair(pep1_index, pep2_index, context, add_precursor=False):
    """
    Generate m/z fragments for a noncovalently bound peptide pair.

    :param pep1_index: (int) Index into the peptide DB for peptide1
    :param pep2_index: (int) Index into the peptide DB for peptide2
    :param context (Searcher) search context
    :param add_precursor (bool) Adds the precursor fragments if True
    :return: (ndarray) fragments numpy array with mz, charge and ion_type
    """
    # generate fragments
    pep1_frags = fragment_linear_peptide(pep1_index, context, add_precursor)
    pep2_frags = fragment_linear_peptide(pep2_index, context, add_precursor)

    # set pep_id to 2 for fragments from peptide 2
    pep2_frags['pep_id'] = 2

    # swap the order of ranges for pep2
    pep2_range0 = pep2_frags['ranges'][:, 0].copy()
    pep2_range1 = pep2_frags['ranges'][:, 1].copy()
    pep2_frags['ranges'][:, 0] = pep2_range1
    pep2_frags['ranges'][:, 1] = pep2_range0

    all_frags = np.concatenate([pep1_frags, pep2_frags])

    all_frags.sort()

    return all_frags


def fragment_crosslinked_peptide(frag_pep_index, cl_pep_index, link_pos,
                                 crosslinker, context, add_precursor=False):
    """
    Generate the fragments for a crosslinked peptide.

    :param frag_pep_index: (int) Index into the peptide DB for the peptide to fragment
    :param cl_pep_index: (int) Index into the peptide DB for the crosslinked peptide
    :param link_pos: (int|list of int) 0-based position of the cross-linker
                if list is given fragments fulfilling any link-site will be generated
    :param crosslinker: (Crosslinker) the crosslinker linking the peptides
    :param context (Searcher) search context
    :param add_precursor: (bool) if True precursor fragments (ions) are added.
     peak (default: 0 - only monoisotopic peaks).
    :return: (ndarray) structured array of fragments
    """
    # generate linear ions
    frag_table = fragment_linear_peptide(peptide_index=frag_pep_index,
                                         context=context, add_precursor=add_precursor)

    # number of possible ions from each terminal
    # for cross-linked peptides the precursor only needs to be added once
    if add_precursor:
        n_ions = np.amax(frag_table["idx"], initial=0) - 1
    else:
        n_ions = np.amax(frag_table["idx"], initial=0)

    link_pos_max = max(link_pos) if isinstance(link_pos, Iterable) else link_pos
    link_pos_min = min(link_pos) if isinstance(link_pos, Iterable) else link_pos

    # some conditions are used multiple times - so prepare these
    mask_n = frag_table["term"] == b'n'
    mask_c = frag_table["term"] == b'c'
    mask_p = frag_table["term"] == b'P'
    mask_n_xl = mask_n & (frag_table["idx"] > link_pos_max)
    mask_c_xl = mask_c & (frag_table["idx"] > (n_ions - link_pos_min))
    mask_xl = mask_n_xl | mask_c_xl | mask_p

    if cl_pep_index is None:
        second_pep_mod_mass = 0
    else:
        # calculate the mass of the 2nd peptide + crosslinker
        second_pep_mass = context.peptide_db.peptide_mass(cl_pep_index)
        second_pep_mod_mass = second_pep_mass + crosslinker.mass

    # add crosslinker modification mass to crosslinker containing fragments
    frag_table["mz"][mask_xl] += second_pep_mod_mass / frag_table["charge"][mask_xl]

    # change linear annotation of these fragments
    frag_table["LN"][mask_xl] = False

    # update the peptide ranges for these fragments
    second_pep_aa_len = context.modified_peptides_aa_lengths[cl_pep_index]
    frag_table["ranges"][:, 1][mask_xl] = [0, second_pep_aa_len]
    # if we have several possible site add new crosslinked fragments that fulfill the other link
    # sites
    if link_pos_min != link_pos_max:

        # copy additional fragments that could be cross-linked
        mask_n_add = (~ mask_n_xl) & mask_n & (frag_table["idx"] > link_pos_min)
        mask_c_add = (~ mask_c_xl) & mask_c & (frag_table["idx"] > (n_ions - link_pos_max))
        mask_add = mask_n_add | mask_c_add

        frag_add = frag_table[mask_add]

        # and turn them into cross-linked fragments

        frag_add["mz"] += second_pep_mod_mass / frag_add["charge"]
        frag_add["LN"] = False
        frag_add["ranges"][:, 1] = [0, second_pep_aa_len]

        # and add them back into the list
        frag_table = np.concatenate((frag_table, frag_add))

    # add crosslinker stub fragments
    stub_fragments_list = []
    peptide_mass = context.peptide_db.peptide_mass(frag_pep_index)
    peptide_aa_len = context.modified_peptides_aa_lengths[frag_pep_index]

    for stub in crosslinker.cleavage_stubs:
        # first add peptide stub fragments
        pep_stub_frags = np.zeros(1, dtype=frag_table.dtype)
        pep_stub_frags['charge'] = 1
        pep_stub_frags['mz'] = peptide_mass + stub.mass + PROTON_MASS
        pep_stub_frags['LN'] = True
        pep_stub_frags['loss'] = b''
        pep_stub_frags['nlosses'] = 0
        pep_stub_frags['stub'] = stub.name
        pep_stub_frags['ion_type'] = b'P'
        pep_stub_frags['term'] = b'P'
        pep_stub_frags['idx'] = peptide_aa_len
        pep_stub_frags['pep_id'] = 1
        pep_stub_frags['ranges'] = [[0, peptide_aa_len], [0, 0]]
        stub_fragments_list.append(pep_stub_frags)

        # then create the b/y-like stub fragments from the crosslinker containing fragments
        stub_fragments = frag_table[(~frag_table["LN"]) & (frag_table["ion_type"] != b'P')]
        # replace second_pep_mod_mass by stub mass
        stub_fragments["mz"] = stub_fragments["mz"] + \
            (stub.mass - second_pep_mod_mass) / stub_fragments['charge']
        # turn them into linear fragments
        stub_fragments["LN"] = True
        stub_fragments["ranges"][:, 1] = [0, 0]
        stub_fragments["stub"] = stub.name
        stub_fragments_list.append(stub_fragments)

    # combine lists
    frag_table = np.concatenate([frag_table] + stub_fragments_list)

    return frag_table


def fragment_linear_peptide(peptide_index, context, add_precursor=False):
    """
    Create fragments for a linear peptide up to max_charge.

    :param peptide_index: (int) Index into the peptide DB
    :param context: (Searcher) search context
    :param add_precursor: (bool) Add the precursor
    :return: (ndarray) Table of fragment ions
        'mz': (np.float64) m/z of the fragment,
        'charge': (np.uint8) charge state of the fragment,
        'LN': (np.bool_) linear fragment,
        'loss': ('<U16') neutral loss name, (Note is '' - loss fragments are generated later)
        'nlosses': (np.uint8) number of losses, (Note is 0 - loss fragments are generated later)
        'stub': ('<U1') crosslinker stub type
        'ion_type': ('<U1') ion type of the fragment, e.g. 'b',
        'term': ('<U1') terminus of the fragment ('n' or 'c'),
        'idx': (np.uint8) fragment number,
        'ranges': (np.uint8, (2, 2))) range over the peptides as 2x2 array:
            (start aa position on first peptide, end aa position on first peptide)
            (start aa position on second peptide, end aa position on second peptide)
                Note: second row will be (0, 0) since this is function creates only linear peptides
    """
    # Get basic fragments from DB
    fragments = context.fragmentation.fragment_peptide(peptide_index, add_precursor)

    return fragments


def fragment_crosslinked_peptide_pair_minimal(pep1_index, pep2_index, link_pos1, link_pos2,
                                              crosslinker, context):
    """
    Generate singly charged fragment mz values, linearities and the masses of the
    attached crosslinker stubs for a crosslinked peptide pair.

    :param pep1_index: (int) Index into the peptide DB for peptide1
    :param pep2_index: (int) Index into the peptide DB for peptide2
    :param link_pos1: (int) 0-based position of the cross-linking site on peptide 1
    :param link_pos2: (int) 0-based position of the cross-linking site on peptide 2
    :param crosslinker: (Crosslinker) the crosslinker linking the peptides
    :param context (Searcher) search context
    :return: (ndarray) mz values, (ndarray) linearity mask, (ndarray) stub masses
    """
    pep1_mz, pep1_linear, pep1_stub_masses = fragment_crosslinked_peptide_minimal(pep1_index,
                                                                                  pep2_index,
                                                                                  link_pos1,
                                                                                  crosslinker,
                                                                                  context)
    pep2_mz, pep2_linear, pep2_stub_masses = fragment_crosslinked_peptide_minimal(pep2_index,
                                                                                  pep1_index,
                                                                                  link_pos2,
                                                                                  crosslinker,
                                                                                  context)

    all_mz = np.concatenate([pep1_mz, pep2_mz])
    all_linear = np.concatenate([pep1_linear, pep2_linear])
    all_stubs = np.concatenate([pep1_stub_masses, pep2_stub_masses])

    return all_mz, all_linear, all_stubs


def fragment_noncovalent_peptide_pair_minimal(pep1_index, pep2_index, context):
    """
    Generate singly charged fragment mz values, linearities (True) and
    stub-masses (0) for a noncovalently bound peptide pair.

    :param pep1_index: (int) Index into the peptide DB for peptide1
    :param pep2_index: (int) Index into the peptide DB for peptide2
    :param context (Searcher) search context
    :return: (ndarray) mz values, (ndarray) linearity mask, (ndarray) zeros
    """
    pep1_mz, pep1_linear, _ = fragment_linear_peptide_minimal(pep1_index, context)
    pep2_mz, pep2_linear, _ = fragment_linear_peptide_minimal(pep2_index, context)

    all_mz = np.concatenate([pep1_mz, pep2_mz])
    all_linear = np.concatenate([pep1_linear, pep2_linear])

    return all_mz, all_linear, np.zeros(len(all_mz))


def fragment_crosslinked_peptide_minimal(frag_pep_index, cl_pep_index, link_pos,
                                         crosslinker, context):
    """
    Generate singly charged fragment mz values, fragment linearities and fragment attached
    crosslinker-stub masses for a crosslinked peptide.

    :param frag_pep_index: (int) Index into the peptide DB for the peptide to fragment
    :param cl_pep_index: (int) Index into the peptide DB for the crosslinked peptide
    :param link_pos: (int|list of int) 0-based position of the cross-linker
                if list is given fragments fulfilling any link-site will be generated
    :param crosslinker: (Crosslinker) the crosslinker linking the peptides
    :param context (Searcher) search context
    :return: (ndarray) mz values, (ndarray) linearity mask, (ndarray) stub_mass
    """
    # Get basic fragment table from cache
    frag_table = context.fragmentation.fragment_peptide_minimal(frag_pep_index,
                                                                add_precursor=False)
    mz = frag_table["mz"].copy()
    linear = np.ones(len(frag_table), bool)

    # number of possible ions from each terminal
    n_ions = np.amax(frag_table["idx"], initial=0)

    link_pos_max = max(link_pos) if isinstance(link_pos, Iterable) else link_pos
    link_pos_min = min(link_pos) if isinstance(link_pos, Iterable) else link_pos

    # some conditions are used multiple times - so prepare these
    mask_n = frag_table["term"] == b'n'
    mask_c = frag_table["term"] == b'c'
    mask_n_xl = mask_n & (frag_table["idx"] > link_pos_max)
    mask_c_xl = mask_c & (frag_table["idx"] > (n_ions - link_pos_min))
    mask_xl = mask_n_xl | mask_c_xl

    if cl_pep_index is None:
        second_pep_mod_mass = 0
    else:
        # calculate the mass of the 2nd peptide + crosslinker
        second_pep_mass = context.peptide_db.peptide_mass(cl_pep_index)
        second_pep_mod_mass = second_pep_mass + crosslinker.mass

    # add crosslinker modification mass to crosslinker containing fragments
    mz[mask_xl] += second_pep_mod_mass

    # change linear annotation of these fragments
    linear[mask_xl] = False

    # if we have several possible site add new crosslinked fragments that fulfill the other link
    # sites
    if link_pos_min != link_pos_max:

        # copy additional fragments that could be cross-linked
        n_add_mask = (~ mask_n_xl) & mask_n & (frag_table["idx"] > link_pos_min)
        c_add_mask = (~ mask_c_xl) & mask_c & (frag_table["idx"] > (n_ions - link_pos_max))
        add_mask = n_add_mask | c_add_mask

        mz_add = mz[add_mask]

        linear_add = linear[add_mask]

        # and turn them into cross-linked fragments
        mz_add += second_pep_mod_mass
        linear_add[:] = False

        # and add them back into the list
        mz = np.concatenate((mz, mz_add))
        linear = np.concatenate((linear, linear_add))

    # add crosslinker stub fragments
    stub_mz_list = []
    stub_mass_list = []
    stub_linear_list = []
    peptide_mass = context.peptide_db.peptide_mass(frag_pep_index)
    # sofar no fragments have stubs - hence stub mass = 0
    stub_mass = np.zeros(len(linear), float)

    for stub in crosslinker.cleavage_stubs:
        # first add peptide stub fragments
        pep_stub_mz = np.array([peptide_mass + stub.mass + PROTON_MASS])
        stub_mz_list.append(pep_stub_mz)
        stub_mass_list.append(np.array([stub.mass]))

        pep_stub_linear = np.ones(1, bool)
        stub_linear_list.append(pep_stub_linear)

        # then create the b/y-like stub fragments from the crosslinker containing fragments
        stub_mz = mz[~linear]
        # replace second_pep_mod_mass by stub mass
        stub_mz += stub.mass - second_pep_mod_mass
        # turn them into linear fragments
        stub_mz_list.append(stub_mz)
        # store the mass of the stub
        stub_mass_list.append(np.full(len(stub_mz), stub.mass))

        stub_linear = np.ones(len(stub_mz), bool)
        stub_linear_list.append(stub_linear)

    # combine lists
    mz = np.concatenate([mz] + stub_mz_list)
    linear = np.concatenate([linear] + stub_linear_list)
    stub_mass = np.concatenate([stub_mass] + stub_mass_list)

    return mz, linear, stub_mass


def fragment_linear_peptide_minimal(peptide_index, context):
    """
    Generate singly charged fragment mz values, linearities (True) and
    stub-masses (0) for a linear peptide.

    :param peptide_index: (int) Index into the peptide DB
    :param context: (Searcher) search context
    :return: (ndarray) mz values, (ndarray) linearity mask, (ndarray) zero
    """
    # Get basic fragments from DB
    basic_table = context.fragmentation.fragment_peptide_minimal(peptide_index, add_precursor=False)

    linear = np.ones(len(basic_table), bool)

    stub_masses = np.zeros(len(basic_table))

    return basic_table['mz'], linear, stub_masses
