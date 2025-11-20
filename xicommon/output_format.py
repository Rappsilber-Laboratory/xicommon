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

"""Module for formatting output fields of the search."""
import numpy as np


def create_mod_str(context, pep_indices):
    """
    Create a semicolon separated string of modifications.

    :param context: (Searcher) Search context
    :param pep_indices: (list or Series) Indices into the peptide DB
    :return: modifications strings
    :rtype: list of str
    """
    mod_names = [''] + [mod.name for mod in context.config.modification.modifications]

    mods = context.peptide_db.peptides[pep_indices]['modifications'].tolist()
    # change cterm position to end of list
    [m.append(m.pop(1)) for m in mods]
    mod_strs = [';'.join([mod_names[i] for i in m if i != 0]) for m in mods]
    return mod_strs


def create_mod_pos_str(context, pep_indices):
    """
    Create a semicolon separated string of modification positions.

    :param context: (Searcher) Search context
    :param pep_indices: (list or Series) Indices into the peptide DB
    :return: modification positions strings
    :rtype: list of str
    """
    mod_arrs = context.peptide_db.peptides['modifications'][pep_indices]
    return_strs = []
    for mods in mod_arrs:
        mod_strs = []
        # check for nterm mod
        if mods[0] != 0:
            mod_strs.append('nterm')

        # get aa modification positions
        mod_pos = np.where(mods[2:] != 0)[0]
        # switch to 1-based
        mod_pos += 1
        # convert to str
        mod_strs += mod_pos.astype(str).tolist()

        # check for cterm mod
        if mods[1] != 0:
            mod_strs.append('cterm')
        return_strs.append(';'.join(mod_strs))

    return return_strs


def create_mod_mass_str(context, pep_indices):
    """
    Create a semicolon separated string of modification masses.

    :param context: (Searcher) Search context
    :param pep_indices: (list or Series) Indices into the peptide DB
    :return: modification masses strings
    :rtype: list of str
    """
    mod_deltas = [''] + [str(mod.mass) for mod in context.config.modification.modifications]

    mods = context.peptide_db.peptides[pep_indices]['modifications'].tolist()
    # change cterm position to end of list
    [m.append(m.pop(1)) for m in mods]
    mod_strs = [';'.join([mod_deltas[i] for i in m if i != 0]) for m in mods]
    return mod_strs
