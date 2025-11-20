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

"""Module for creating synthetic (in-silico) mass spectra of peptides."""
from pyteomics import mgf
from xicommon.fragment_peptides import fragment_crosslinked_peptide, fragment_linear_peptide
from xicommon import const
import numpy as np
from numpy.lib import recfunctions
from xicommon.fragmentation import include_losses, spread_charges


def add_isotopes(fragments, isotopes):
    """
    Add a set of isotopic masses to the list of masses.

    :param fragments: monoisotopic masses
    :param isotopes:number of isotopes to add
    :return:
    """
    if isotopes > 0:
        # generate isotope peaks
        isotopes_ar = np.arange(1, isotopes + 1)
        isotope_mz = fragments["mz"].reshape(-1, 1) + \
            (isotopes_ar * const.C12C13_MASS_DIFF) / fragments["charge"].reshape(-1, 1)

        # recreate fragment table to append isotope data
        # init table
        iso_table = np.empty(isotopes * fragments.shape[0],
                             dtype=fragments.dtype)

        # stack multidimensional isotope_mz array to 1-dimension
        iso_table["mz"] = np.hstack(isotope_mz)

        # annotation data does not change, simply copy the existing rows
        iso_table["charge"] = np.repeat(fragments["charge"], isotopes)
        iso_table["LN"] = np.repeat(fragments["LN"], isotopes)
        iso_table["loss"] = np.repeat(fragments["loss"], isotopes)
        iso_table["nlosses"] = np.repeat(fragments["nlosses"], isotopes)
        iso_table["ion_type"] = np.repeat(fragments["ion_type"], isotopes)
        iso_table["term"] = np.repeat(fragments["term"], isotopes)
        iso_table["idx"] = np.repeat(fragments["idx"], isotopes)

        # create the new array with the number of isotopes
        # for each row in the all_fragments table
        # first add zeroes for the monoisotopic peaks
        # then add the isotope count
        isotopes_data = np.concatenate((np.zeros(fragments.shape[0]),
                                        np.repeat(isotopes_ar,
                                                  fragments.shape[0])))

        # combine data
        fragments = np.append(fragments, iso_table)

        # add the newly created isotope annotation
        fragments = recfunctions.append_fields(fragments,
                                               names="n_isotopes",
                                               data=isotopes_data,
                                               dtypes=[np.uint8],
                                               usemask=False)

    return fragments


def create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2, charge,
                              crosslinker, context, isotopes=0, precursor_delta=0):
    """
    Generate a theoretical crosslinked spectrum.

    Fragments from peptide 1 get assigned intensity of 100.
    Fragments from peptide 2 get assigned intensity of 50.

    :param pep1_index: (int) Index into the peptide DB for peptide 1
    :param pep2_index: (int) Index into the peptide DB for peptide 2
    :param link_pos1: (int) 0-based position of the cross-linking site on peptide 1
    :param link_pos2: (int) 0-based position of the cross-linking site on peptide 2
    :param charge: (int) precursor charge
    :param crosslinker: (Crosslinker) the crosslinker linking the peptides
    :param context: (Searcher) search context
    :param precursor_delta: (int) adds a difference to the precursor m/z of the spectrum
    :param isotopes: (int) number of additional isotope peaks to generate for each
              monoisotopic peak (default: 0 - only monoisotopic peaks).
    :return: (str) path to mgf file
    """
    add_precursor = context.config.fragmentation.add_precursor
    pep1_mass = context.peptide_db.peptide_mass(pep1_index)

    pep_sequences = context.peptide_db.mod_pep_sequence([pep1_index, pep2_index]).astype(str)

    if crosslinker is not None:
        cross_linker_mod_mass = crosslinker.mass
        pep2_mass = context.peptide_db.peptide_mass(pep2_index)

        pep1_frags = fragment_crosslinked_peptide(frag_pep_index=pep1_index,
                                                  cl_pep_index=pep2_index,
                                                  link_pos=link_pos1,
                                                  crosslinker=crosslinker, context=context,
                                                  add_precursor=add_precursor)

        pep2_frags = fragment_crosslinked_peptide(frag_pep_index=pep2_index,
                                                  cl_pep_index=pep1_index,
                                                  link_pos=link_pos2,
                                                  crosslinker=crosslinker, context=context)

        unique_pep_id = '{}_{}-{}_{}:{}'.format(
            pep_sequences[0], pep_sequences[1], link_pos1, link_pos2, charge)
    else:
        # linear
        cross_linker_mod_mass = 0
        pep1_frags = fragment_linear_peptide(peptide_index=pep1_index,
                                             context=context, add_precursor=add_precursor)
        pep2_mass = 0
        unique_pep_id = '{}:{}'.format(pep_sequences[0], charge)
        if pep2_index != -1:
            # noncovalent
            pep2_frags = fragment_linear_peptide(peptide_index=pep2_index,
                                                 context=context, add_precursor=add_precursor)
            pep2_mass = context.peptide_db.peptide_mass(pep2_index)
            unique_pep_id = '{}&{}:{}'.format(pep_sequences[0], pep_sequences[1], charge)

    # add losses
    pep1_frags = include_losses(pep1_frags, [pep1_index], context)
    # add isotopes
    pep1_frags = add_isotopes(pep1_frags, isotopes)
    pep1_frags = spread_charges(fragment_table=pep1_frags, context=context, max_charge=charge)
    # create peak list with intensity 100 for all pep1 peaks
    pep1_frag_peak_list = [(m, 100) for m in pep1_frags['mz']]

    if pep2_index != -1:
        pep2_frags = include_losses(pep2_frags, [pep2_index], context)
        pep2_frags = add_isotopes(pep2_frags, isotopes)
        pep2_frags = spread_charges(pep2_frags, context, charge)
        pep2_frag_peak_list = [(m, 50) for m in pep2_frags['mz']]
        fragment_peak_list = pep1_frag_peak_list + pep2_frag_peak_list
    else:
        fragment_peak_list = pep1_frag_peak_list

    precursor_mass = pep1_mass + pep2_mass + cross_linker_mod_mass + precursor_delta
    precursor_mz = (precursor_mass / charge) + const.PROTON_MASS

    fragment_peak_list.sort(key=lambda x: x[0])

    mz_array = [x[0] for x in fragment_peak_list]
    int_array = [x[1] for x in fragment_peak_list]
    params = {
        'TITLE': unique_pep_id,
        'PEPMASS': precursor_mz,
        'CHARGE': '{}+'.format(charge),
    }

    spectrum_dict = {
        'm/z array': mz_array,
        'intensity array': int_array,
        'params': params
    }

    return spectrum_dict


def create_synthetic_spectra_mgf(synthetic_spectra, out_path):
    """
    Create an MGF from synthetic spectra.

    Wrapper around pyteomics write MGF function.
    :param synthetic_spectra: list of synthetic spectra dicts
    :param out_path: (str) folder where to save the MGF file in
    """
    mgf.write(synthetic_spectra, out_path, file_mode='w').close()


def create_synthetic_load_test_mgf(peptides, filename, context, n_spectra=10000, max_charge=6,
                                   isotopes=0):
    """
    Create a synthetic spectra MGF from randomly selected peptides from a FASTA file.

    :param peptides: (array) string array of peptide sequences
    :param filename: (str) filename for output file
    :param context: (Searcher) Search context
    :param n_spectra: (int) number of spectra to generate
    :param max_charge: (int) max_charge
    :param isotopes: (int) number of additional isotope peaks to generate for each monoisotopic
     peak (default: 0 - only monoisotopic peaks).
    """
    pep1_indices = np.random.choice(peptides.size, n_spectra)
    pep2_indices = np.random.choice(peptides.size, n_spectra)

    spectra = []
    for pep1_index, pep2_index in zip(pep1_indices, pep2_indices):
        pep1_aa_len = context.modified_peptides_aa_lengths[pep1_index]
        pep2_aa_len = context.modified_peptides_aa_lengths[pep2_index]
        link_pos1 = pep1_aa_len - 1
        link_pos2 = pep2_aa_len - 1
        charge = np.random.randint(3, max_charge+1)
        spec = create_synthetic_spectrum(pep1_index=pep1_index, pep2_index=pep2_index,
                                         link_pos1=link_pos1, link_pos2=link_pos2,
                                         charge=charge, crosslinker=context.config.crosslinker[0],
                                         context=context, isotopes=isotopes)

        spectra.append(spec)

    create_synthetic_spectra_mgf(spectra, filename)
