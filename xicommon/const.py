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

"""Module providing constants and regular expressions."""
import re
import sys


class _const:
    # xisearch version
    VERSION = "2.0.2"

    PROTON_MASS = 1.007276466879
    C12C13_MASS_DIFF = 1.0033548
    # mass of glycine (smallest amino acid)
    AMINO_ACID_MIN_MASS = 57.02147
    # according to Senko et al. (doi: 10.1016/1044-0305(95)00017-8)
    AVG_AMINO_ACID_MASS = 111.1254
    # Create pattern used to split a modX peptide sequence string into list of modified amino acids,
    # discarding n- and c-terminal modifications
    # example: "nterm-PEmod1PTImod2DEmod3A-cterm"
    # matches: ['P', 'E', 'mod1P', 'T', 'I', 'mod2D', 'E', 'mod3A']
    # this is used with re.findall and seemingly if the regular expression contains capturing groups
    # only these will be kept.
    PEPTIDE_TO_AMINO_ACID = re.compile(b'([^A-Z\\-]*[A-Z])')
    # Create pattern used to split peptide sequence string into list of
    # modified amino acids.
    # Matches non-uppercase characters [^A-Z]* before the uppercase amino acid
    # in one letter code: [A-Z]
    # Can also match c-terminal modifications: non-uppercase with leading -
    # at the end of the string:
    # (?:-[^A-Z]*$)
    # example: "nterm-PEmod1PTImod2DEmod3A-cterm"
    # matches: ['nterm-P', 'E', 'mod1P', 'T', 'I', 'mod2D', 'E', 'mod3A-cterm']
    MODIFIED_AA_PATTERN = re.compile(b"[^A-Z]*[A-Z](?:-[^A-Z]*$)?")
    H_MASS = 1.007825032241
    H2O_MASS = 18.0105647

    # as const is overwriten by _const the module __file__ variable would disapear.
    # so it is also saved into the class _const
    __file__ = __file__

    class ConstError(TypeError):
        pass

    # overwrite the __setattr__ method to raise an error if a variable is overwritten
    def __setattr__(self, name, value):
        if name in self.__dict__ or name in self.__class__.__dict__:
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name] = value


# overwrite the module const with the class _const so that we can actually protect attributes
# from being changed
sys.modules[__name__] = _const()
