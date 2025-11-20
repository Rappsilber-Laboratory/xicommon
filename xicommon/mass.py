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

from pyteomics.cmass import calculate_mass, std_aa_comp, std_ion_comp
from xicommon.const import PROTON_MASS
from xicommon.utils import get_chunks
import numpy as np

# Generate lookup table mapping ASCII values to AA masses.
aa_masses = np.empty(256)
aa_masses[0] = 0  # assume strings are 0-terminated and padded.
aa_masses[1:] = np.nan  # make invalid AAs result in an invalid mass.

# calculate masses from compositions (std_aa_mass dict contains wrong entry for 'O'
for AA, comp in std_aa_comp.items():
    if AA not in ['-OH', 'H-']:
        aa_masses[ord(AA)] = calculate_mass(composition=comp)
ion_masses = np.empty(256)
ion_masses[:] = np.nan
# Generate lookup table for ion mass differences

for ion_type, composition in std_ion_comp.items():
    if len(ion_type) == 1:
        ion_masses[ord(ion_type)] = calculate_mass(composition=composition)

# add entry for precursor in ion composition dict
ion_masses[ord("P")] = 0

# Precalculate terminal mass
nterm_mass = calculate_mass(formula='H')
cterm_mass = calculate_mass(formula='OH')
unmodified_termini_mass = nterm_mass + cterm_mass


def mass(sequences=None, sequence_indices=None, modifications=None,
         losses=None, ion_types=None, charges=None, config=None):
    """
    Calculate masses of sequences and/or modifications.

    :param sequences: (ndarray, bytes, ndim=1), optional
        Unique sequence strings.

    :param sequence_indices: (ndarray, int, ndim=1), optional
        Sequence indices to use for each variation. This allows the same sequences to be used
        multiple times to minimise the size of the necessary string array. If not given, a
        single variation per sequence is assumed.

    :param modifications: (ndarray, uint8, ndim=2), optional
        Modification locations. Shape should be N x (L + 2), where N is the number of
        variations, and L is the maximum sequence length. In the second axis, index 0 is the
        n-terminus and 1 is the c-terminus. Indices 2 onwards are the amino acids of the
        sequence. Values should be zero to indicate no modification, or one plus an index
        into the config modification list, to indicate a modification at that location.

    :param losses: (ndarray, uint8, ndim=2), optional
        Loss counts. Shape should be N x L, where N is the number of variations, and L is the
        number of configured losses. Values are the number of losses of each loss type. The
        locations of the losses are not represented or considered.

    :param ion_types: (ndarray, str, ndim=1), optional
        Ion types. Names should match those used by pyteomics, see std_ion_comp.

    :param charges: (ndarray, int, ndim=1)
        Ion charges. Length should match the number of variations.

    :param config: (Config)
        Search configuration, referred to for modification and loss masses.
    """
    # Get number of masses to be calculated
    if sequence_indices is not None:
        num_masses = len(sequence_indices)
    elif sequences is not None:
        num_masses = len(sequences)
    elif modifications is not None:
        num_masses = len(modifications)
    else:
        raise ValueError("Either sequences or modifications must be given")

    if sequences is not None:
        # Allocate output array
        masses = np.empty(num_masses, np.float64)

        # Cast byte strings to uint8 array
        sequence_chars = sequences.view(np.uint8).reshape(len(sequences), -1)
    else:
        masses = np.zeros(num_masses, np.float64)

    # Process in chunks to minimise memory usage
    for chunk_start, chunk_length, chunk in get_chunks(num_masses, 1000):

        if sequences is not None:
            if sequence_indices is not None:
                chunk_sequence_chars = sequence_chars[sequence_indices[chunk]]
            else:
                chunk_sequence_chars = sequence_chars[chunk]

            # Get sequence masses using lookup table
            masses[chunk] = aa_masses[chunk_sequence_chars].sum(axis=1)

        # Add masses of unmodified termini
        masses[chunk] += unmodified_termini_mass
        if modifications is not None:
            # Get masses of configured modifications
            mod_deltas = np.array([0] + [mod.mass for mod in config.modification.modifications])
            # Add modification masses
            masses[chunk] += mod_deltas[modifications[chunk]].sum(axis=1)

        if losses is not None:
            # Get mass deltas of configured losses
            loss_deltas = np.array([loss.mass for loss in config.fragmentation.losses])
            # Subtract loss masses
            masses[chunk] -= (losses[chunk] * loss_deltas).sum(axis=1)

        if ion_types is not None:
            # Add ion mass differences
            masses[chunk] += ion_mass(ion_types[chunk])

        if charges is not None:
            # Add proton masses
            masses[chunk] += charges[chunk] * PROTON_MASS

    return masses


def ion_mass(ion_types):
    """Get mass differences for all ion types in input."""
    ion_type_codes = np.char.encode(ion_types, 'ascii').view(np.uint8)
    return ion_masses[ion_type_codes]


def ion_mass_byte(ion_types):
    """Get mass differences for all ion types in input."""
    ion_type_codes = np.array(ion_types).view(np.uint8)
    return ion_masses[ion_type_codes]
