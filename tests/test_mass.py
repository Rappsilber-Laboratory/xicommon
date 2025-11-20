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

from xicommon.config import (Config,
                             ModificationConfig,
                             Modification,
                             FragmentationConfig,
                             Loss)
from xicommon.modifications import modified_sequence_strings
from pyteomics.mass import calculate_mass
from xicommon.mass import mass
from xicommon import const
import numpy as np
import pytest


def test_mass():

    # Test cases

    config = Config(
        modification=ModificationConfig(
            modifications=[
                Modification(name='ox', specificity=['M'], composition="O1", type='variable'),
                Modification(name='cm', specificity=['C'], composition="C2H3O1N1", type='variable'),
                Modification(name='acetyl-', specificity=['X'], composition="C2H2O1",
                             type='variable'),
                Modification(name='-amide', specificity=['X'], composition="N1H1O-1",
                             type='variable'),
            ]),
        fragmentation=FragmentationConfig(
            losses=[
                Loss(name='H20', composition="H2O1", specificity=['S', 'T', 'E', 'D']),
            ]))

    sequences = np.array([b'PEPTIDE', b'KING', b'ELVIS', b'LACK', b'MACKED'])

    sequence_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    modifications = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # PEPTIDE
        [3, 4, 0, 0, 0, 0, 0, 0, 0],  # acetyl-PEPTIDE-amide
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # KING
        [3, 0, 0, 0, 0, 0, 0, 0, 0],  # acetyl-KING
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # ELVIS
        [0, 4, 0, 0, 0, 0, 0, 0, 0],  # ELVIS-amide
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # LACK
        [0, 0, 0, 0, 2, 0, 0, 0, 0],  # LAcmCK
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # MACKED
        [3, 4, 1, 0, 2, 0, 0, 0, 0],  # acetyl-oxMAcmCKED-amide
    ], np.uint8)

    losses = np.array([[0], [3], [0], [0], [0], [1], [0], [0], [0], [2]], np.uint8)

    ion_types = np.array(['c', 'z', 'a', 'b', 'y', 'c', 'z', 'a', 'b', 'y'])

    charges = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    # Generate expected values using pyteomics

    modified_peptides = np.empty(len(modifications), dtype=[('sequence_index', np.intp),
                                                            ('modifications', np.uint8, (9,))])
    modified_peptides['sequence_index'] = sequence_indices
    modified_peptides['modifications'] = modifications

    modified_sequences = modified_sequence_strings(sequences, modified_peptides, config)

    sequence_masses = np.array([calculate_mass(sequence=modified_sequences[i].decode('ascii'),
                                               ion_type=ion_types[i], charge=int(charges[i]),
                                               aa_comp=config.modification.compositions)
                                for i in range(len(modified_sequences))])
    # c and n-terminal modifications are not compatible with pyteomis - so need to
    # correct these
    charge_zero = charges == 0
    nterm = modified_peptides['modifications'][:, 0] != 0
    sequence_masses[np.where(nterm & charge_zero)] += const.H_MASS
    sequence_masses[np.where(nterm & ~charge_zero)] += \
        const.H_MASS/charges[np.where(nterm & ~charge_zero)]

    cterm = modified_peptides['modifications'][:, 1] != 0
    sequence_masses[np.where(cterm & charge_zero)] += const.H2O_MASS - const.H_MASS
    sequence_masses[np.where(cterm & ~charge_zero)] += \
        (const.H2O_MASS - const.H_MASS)/charges[np.where(cterm & ~charge_zero)]

    # calculate_mass returns m/z, not mass
    sequence_masses[charges != 0] *= charges[charges != 0]

    loss_masses = losses * np.array([loss.mass for loss in config.fragmentation.losses])

    expected_masses = sequence_masses - loss_masses.sum(axis=1)

    # Check our result against pyteomics result

    masses = mass(sequences, sequence_indices, modifications,
                  losses, ion_types, charges, config)

    np.testing.assert_almost_equal(masses, expected_masses, decimal=4)

    with pytest.raises(ValueError):
        masses = mass(None, None, None,
                      losses, ion_types, charges, config)
