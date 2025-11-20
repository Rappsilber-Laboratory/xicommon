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
Lightweight database implementations for annotation and testing.

This module provides simplified, in-memory database classes for peptide and fragment
storage that don't depend on search-specific code. These are used by MockContext
and the annotator.
"""

import numpy as np
from xicommon.mass import mass
from xicommon.const import PROTON_MASS
from xicommon.modifications import modified_sequence_strings


class SimplePeptideDatabase:
    """
    Lightweight in-memory peptide database for annotation/testing.

    Unlike the full PeptideDatabase in the search project, this version:
    - Stores everything in memory (no file-based storage)
    - Simpler implementation
    - No dependency on search infrastructure
    """

    def __init__(self, context):
        """
        Initialize SimplePeptideDatabase.

        :param context: Context object with config, modified_peptides, and
                        unmodified_peptide_sequences
        """
        self.config = context.config
        self.peptides = context.modified_peptides
        self.unmodified_sequences = context.unmodified_peptide_sequences

        # Build mass table in memory
        self.ids = np.arange(self.peptides.size)
        self.masses = mass(
            self.unmodified_sequences[self.peptides['sequence_index']],
            modifications=self.peptides['modifications'],
            config=self.config
        )

        # Create reverse index for fast lookups
        self.reverse = np.argsort(self.ids)
        self.delta_masses = np.array([])
        self.num_peptides = len(self.peptides)

    def peptide_mass(self, peptide_index):
        """
        Return the mass of the peptide(s) with the given index.

        :param peptide_index: (int|ndarray) Single index or array of indices
        :return: Mass of the peptide(s)
        :rtype: single float64 or ndarray (float64)
        """
        mass_index = self.reverse[peptide_index]
        return self.masses[mass_index]

    def unmod_pep_sequence(self, peptide_index):
        """
        Return unmodified peptide sequence(s).

        :param peptide_index: (int|ndarray) Single index or array of indices
        :return: unmodified peptide sequence(s)
        :rtype: single byte string or ndarray (bytes)
        """
        return self.unmodified_sequences[
            self.peptides['sequence_index'][peptide_index]
        ]

    def mod_pep_sequence(self, peptide_index, mod_peptide_syntax=None):
        """
        Return modified peptide sequences.

        :param peptide_index: (ndarray) array of indices into the peptide DB
        :param mod_peptide_syntax: (str) syntax to use for modified peptides returned
            (defaults to config.mod_peptide_syntax)
        :return: modified peptide sequences
        :rtype: ndarray (bytes)
        """
        if mod_peptide_syntax is None:
            mod_peptide_syntax = self.config.mod_peptide_syntax

        mod_sequences = modified_sequence_strings(
            self.unmodified_sequences,
            self.peptides[peptide_index],
            self.config,
            mod_peptide_syntax
        )

        # correct linear sequences
        mod_sequences[peptide_index == -1] = b''

        return mod_sequences

    def lookup(self, limits, return_indices=False, return_values=False, return_counts=False):
        """
        Fast mass-based lookup.

        :param limits: Mass range limits for lookup
        :param return_indices: Return indices of matches
        :param return_values: Return values of matches
        :param return_counts: Return counts of matches
        :return: Lookup results
        """
        from xicommon.cython import fast_lookup
        return fast_lookup(
            self.masses, self.ids, limits,
            return_indices, return_values, return_counts
        )


class SimpleFragmentDatabase:
    """
    Lightweight in-memory fragment database for annotation.

    Unlike the full FragmentDatabase in the search project, this version:
    - Builds fragments in memory
    - Simpler construction
    - No dependency on search infrastructure
    """

    def __init__(self, context):
        """
        Initialize SimpleFragmentDatabase.

        :param context: Context object with sequences, modified_peptides, and config
        """
        self.context = context
        self.sequences = context.unmodified_peptide_sequences
        self.modified_peptides = context.modified_peptides
        self.num_peptides = len(self.modified_peptides)

        # Get stub masses from config if crosslinker defined
        if hasattr(context.config, 'crosslinker'):
            self.delta_masses = np.array([
                s.mass for xl in [
                    allxl for allxl in context.config.crosslinker
                    if hasattr(allxl, "stubs")
                ] for s in xl.stubs
            ])
        else:
            self.delta_masses = np.array([])

        # Build fragment table
        self._build_fragment_table()

    def _build_fragment_table(self):
        """Generate fragment table using fragmentation from common."""
        from xicommon.fragmentation import fragment_ions

        # Generate fragments for all peptides
        masses_list = []
        ids_list = []

        generator = fragment_ions(
            self.sequences,
            self.modified_peptides[~self.modified_peptides['linear_only']],
            self.context,
            add_precursor=True
        )

        for term, ion, loss, loss_count, sites, masses in generator:
            peptide_indices = sites['peptide_index']
            masses_list.append(masses + PROTON_MASS)
            ids_list.append(peptide_indices)

        # Combine and sort by mass
        if masses_list:
            self.masses = np.concatenate(masses_list)
            self.ids = np.concatenate(ids_list)

            # Sort by mass for fast lookups
            sort_order = np.argsort(self.masses)
            self.masses = self.masses[sort_order]
            self.ids = self.ids[sort_order]
        else:
            self.masses = np.array([])
            self.ids = np.array([])

        self.num_fragments = len(self.masses)

    def lookup(self, limits, return_indices=False, return_values=False, return_counts=False):
        """
        Fast mass-based fragment lookup.

        :param limits: Mass range limits for lookup
        :param return_indices: Return indices of matches
        :param return_values: Return values of matches
        :param return_counts: Return counts of matches
        :return: Lookup results
        """
        from xicommon.cython import fast_lookup
        return fast_lookup(
            self.masses, self.ids, limits,
            return_indices, return_values, return_counts
        )
