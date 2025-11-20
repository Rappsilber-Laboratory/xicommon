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

"""Configuration file for running xisearch."""
import json
import yaml
import re
import copy
from memoized_property import memoized_property
from pyteomics import cmass

# Unique sentinel, used to allow None to be a valid default for a setting.
NO_DEFAULT = object()


def stringhash(s):
    """
    Create a stable hash code for strings.

    When run in debugger hash(str) seems not give a stable result -
    as after restarting the debugger hash(str) with exactly the same string
    gives a different value.
    """
    return hash(tuple(ord(x) for x in s))


class Setting:
    """A setting supported by the config system."""

    def __init__(self, type, default=NO_DEFAULT, valid_values=None, required=True, max_value=None):
        """
        Initialise the Setting.

        :param type: Python type expected for this setting.
        :param default: Default value for this setting.
        :param valid_values: Tuple of accepted values, re pattern, or None to accept any value.
        :param max_value: Maximum value of setting (for float or int types)
        """
        self.type = type
        self.valid_values = valid_values
        if max_value is not None and not any([issubclass(self.type, int),
                                              issubclass(self.type, float)]):
            raise TypeError("max_value is only supported for int and float type.")
        self.max_value = max_value
        if default is not NO_DEFAULT:
            try:
                self.default = self.accept(default)
            except TypeError:
                raise TypeError("Default '%s' is not valid and could not be coerced "
                                "into the expected type (%s)" % (repr(default),
                                                                 repr(self.type))) from None
            self.required = False
        else:
            self.required = required

    def accept(self, value):
        """
        Check if a value has a value is valid.

        :param value: (mixed) value to check
        :return: (bool) True if valid, else False
        """
        coerced_value = self.coerce(value)
        if self.max_value is not None and coerced_value >= self.max_value:
            raise ValueError(f'{coerced_value} is above max_value({self.max_value})!')
        if self.valid_values is not None:
            if isinstance(self.valid_values, re.Pattern):
                if self.valid_values.match(coerced_value) is None:
                    raise ValueError(f'{coerced_value} is not valid!'
                                     f' Valid values need to match: {self.valid_values.pattern}')
            elif coerced_value not in self.valid_values:
                raise ValueError(f'{coerced_value} is not valid!'
                                 f' Valid values are: {self.valid_values}')
        return coerced_value

    def coerce(self, value):
        """
        Coerce a value into the correct type.

        :param value: (mixed) value to coerce
        :return: (mixed) coerced value
        """
        if isinstance(value, self.type):
            return value
        try:
            if isinstance(value, dict):
                return self.type(**value)
            elif value in self.type.__dict__:
                return self.type.__dict__[value]
            else:
                return self.type(value)
        except ValueError:
            raise TypeError from None

    def hash(self, value):
        """
        Create a hash for a value.

        :param value: (mixed) value to create hash for
        :return: (str) hash of the value
        """
        if issubclass(self.type, ConfigGroup):
            return value.hash()
        elif issubclass(self.type, str):
            return stringhash(value)
        else:
            return hash(value)


class ListSetting(Setting):
    """A Setting with a list of values supported by the config system."""

    def accept(self, values):
        """
        Check if all elements of the ListSetting have valid values.

        :param values: (list) values to check
        :return: (bool) True if valid, else False
        """
        return_list = []

        if not isinstance(values, list):
            if isinstance(self.type, Setting):
                value = self.type.accept(values)
                return_list.append(value)
            else:
                value = super().accept(values)
                return_list.append(value)
        else:
            # if type is a ListSetting again - we make some assumptions
            if isinstance(self.type, ListSetting):
                # if we have a list of lists all is fine
                if any([isinstance(v, list) for v in values]):
                    for value in values:
                        value = self.type.accept(value)
                        return_list.append(value)
                else:
                    # we have just a single list - assume
                    # the list is actually the first entry in the outer list
                    value = self.type.accept(values)
                    return_list.append(value)

            # if the "type" is actually a Setting then use that to convert the value
            elif isinstance(self.type, Setting):
                for value in values:
                    value = self.type.accept(value)
                    return_list.append(value)
            else:
                for value in values:
                    value = super().accept(value)
                    return_list.append(value)

        return return_list

    def hash(self, value):
        """
        Create a hash.

        :return: (int) hash
        """
        if isinstance(self.type, Setting):
            return hash(tuple(self.type.hash(x) for x in value))
        elif issubclass(self.type, ConfigGroup):
            return hash(tuple(x.hash() for x in value))
        elif issubclass(self.type, str):
            return hash(tuple(stringhash(x) for x in value))
        else:
            return hash(tuple(value))


class ConfigMeta(type):
    """Metaclass used to define configuration groups."""

    def __new__(cls, name, bases, attributes):
        """Create a new instance."""
        settings = {k: a for k, a in attributes.items() if isinstance(a, Setting)}
        others = {k: a for k, a in attributes.items() if k not in settings}
        defaults = {k: s.default for k, s in settings.items() if hasattr(s, 'default')}
        required = set([k for k, s in settings.items() if s.required])
        new_attributes = dict(_settings=attributes, _defaults=defaults, _required=required,
                              **others)
        return type.__new__(cls, name, bases, new_attributes)


class ConfigGroup(metaclass=ConfigMeta):
    """Base class for configuration groups."""

    def __init__(self, **kwargs):
        """Initialise the ConfigGroup."""
        self._values = {}
        for key, value in kwargs.items():
            if key not in self._settings:
                raise KeyError("Unknown setting '%s'" % key)
            setattr(self, key, value)

        # transfer defaults to those values that are not set explicitly
        for k, v in self._defaults.items():
            if k not in kwargs.keys():
                setattr(self, k, copy.deepcopy(self._defaults[k]))

        for setting in self._required:
            if setting not in kwargs.keys():
                raise AttributeError("'%s' is required but not defined" % setting) from None

    def __setattr__(self, key, value):
        """Set the value of a Setting."""
        if key.startswith('_') or key not in self._settings:
            super(ConfigGroup, self).__setattr__(key, value)
            return
        setting = self._settings[key]
        try:
            self._values[key] = setting.accept(value)
        except TypeError:
            raise TypeError("Value '%s' is not valid for '%s' and could not be coerced "
                            "into the expected type (%s)" % (repr(value), key,
                                                             repr(setting.type))) from None
        except ValueError:
            raise ValueError("Value '%s' is not valid for '%s'" % (repr(value), key)) from None

    def __contains__(self, key):
        """Check if a Setting is configured in the ConfigGroup."""
        return key in self._settings

    def __getattr__(self, key):
        """Get the value for a Setting."""
        if key.startswith('_') or key not in self._settings:
            return super(ConfigGroup, self).__getattr__(key)
        elif key in self._values:
            return self._values[key]
        else:
            raise AttributeError(key)

    def __eq__(self, other):
        """Check if two ConfigGroups are equal."""
        if type(other) is type(self):
            return vars(self) == vars(other)
        return False

    def hash(self):
        """
        Create a hash.

        :return: (int) hash
        """
        value_hashes = [(stringhash(name), self._settings[name].hash(value))
                        for name, value in self._values.items()]
        return hash(frozenset(value_hashes))

    @classmethod
    def from_json(cls, json_string):
        """Create a ConfigGroup from a JSON string."""
        args = json.loads(json_string)
        return cls(**args)

    @classmethod
    def from_yaml(cls, yaml_string):
        """Create a ConfigGroup from a YAML string."""
        args = yaml.safe_load(yaml_string)
        return cls(**args)

    @staticmethod
    def _list_elements_to_dict(values_lst, excl_defaults=True):
        """
        Convert a list of ConfigGroups/values to a list of dictionaries.

        :param values_lst: (list) list of ConfigGroups/values
        :param excl_defaults: (bool) exclude default values
        :return: (list) list of dictionary representations of the ConfigGroups/values
        """
        return_lst = []
        for element in values_lst:
            if isinstance(element, ConfigGroup):
                return_lst.append(element.to_dict(excl_defaults=excl_defaults))
            elif isinstance(element, list):
                return_lst.append(ConfigGroup._list_elements_to_dict(element))
            else:
                return_lst.append(element)
        return return_lst

    def to_dict(self, excl_defaults=True):
        """
        Convert the ConfigGroup to a dictionary.

        :param excl_defaults: (bool) exclude default values
        :return: (dict) dictionary representation of the ConfigGroup
        """
        values = {}
        for k in self._values:
            # convert ConfigGroups to dictionaries recursively
            if isinstance(self._values[k], ConfigGroup):
                value_tmp = self._values[k].to_dict(excl_defaults=excl_defaults)
            # convert lists of ConfigGroups to list of dictionaries recursively
            elif isinstance(self._values[k], list):
                value_tmp = ConfigGroup._list_elements_to_dict(self._values[k],
                                                               excl_defaults=excl_defaults)
            # collect values
            else:
                value_tmp = self._values[k]

            # only keep if not empty:
            if value_tmp is not None:
                # only keep if not default value (if set)
                if k not in self._defaults or self._defaults[k] != value_tmp or not excl_defaults:
                    values[k] = value_tmp
            else:
                pass

        if len(values) == 0:
            return None
        else:
            return values

    def to_json(self, excl_defaults=True):
        """Convert the ConfigGroup to a JSON string."""
        return json.dumps(self.to_dict(excl_defaults=excl_defaults))

    def write(self, file_name, excl_defaults=True):
        """Write the ConfigGroup to a JSON file."""
        with open(file_name, "w") as outfile:
            json.dump(self.to_dict(excl_defaults=excl_defaults), outfile, indent='\t')

    def write_yaml(self, file_name, excl_defaults=True):
        """Write the ConfigGroup to a YAML file."""
        with open(file_name, "w") as outfile:
            yaml.dump(self.to_dict(excl_defaults=excl_defaults), outfile)


class ToleranceContainer():
    """Mixin class for ConfigGroups that contain tolerances."""

    _re_ms_tol = re.compile(r'^[\-\+]?[0-9]+(?:.[0-9]+)?\s*(?:ppm|th|da)$', re.IGNORECASE)

    def parse_ms_tol(self, str_tol):
        """Parse a tolerance string into a float value and unit string."""
        re_ms_tol = re.compile(r"(-?[0-9.]+)\s*(da|th|ppm)", re.IGNORECASE)
        tol, unit = re_ms_tol.search(str_tol).groups()
        return float(tol), unit

    def translate_ms_tol(self, str_tol):
        """Translate a tolerance string into numeric atol/rtol values."""
        atol, rtol = 0, 0
        tol, unit = self.parse_ms_tol(str_tol)

        if unit.lower() == 'da' or unit.lower() == 'th':
            atol = tol
        elif unit.lower() == 'ppm':
            rtol = tol * 1e-6
        else:
            raise ValueError('MS tolerance must be given in ppm, da, or th.')

        return atol, rtol

    def initialise_tolerances(self, **kwargs):
        """Initialise the MS1 and MS2 tolerances."""

        self.ms1_atol, self.ms1_rtol = self.translate_ms_tol(self.ms1_tol)
        self.ms2_atol, self.ms2_rtol = self.translate_ms_tol(self.ms2_tol)

        if self.ms1_atol < 0 or self.ms1_rtol < 0:
            raise ValueError("MS1 error must be set to a positive value!")
        if self.ms2_rtol < 0 or self.ms2_atol < 0:
            raise ValueError("MS2 error must be set to a positive value!")


class Stub(ConfigGroup):
    """Stub configuration."""

    """Name (only single lowercase char allowed)"""
    name = Setting(str, valid_values=re.compile('^[a-zA-Z0-9]$'))

    """Mass in Dalton"""
    mass = Setting(float)

    """Other stub(s) this is connected to"""
    pairs_with = ListSetting(str, default=[])


class Crosslinker(ConfigGroup):
    """Crosslinker configuration."""

    """Name"""
    name = Setting(str, valid_values=re.compile('^.{1,39}$'))

    """Mass in Dalton"""
    mass = Setting(float)

    """Specificity of crosslinker reactivity as list of specificities for each crosslinker end"""
    specificity = ListSetting(ListSetting(str))

    """Weights for crosslinker specificity (higher values are better)"""
    specificity_bonus = ListSetting(ListSetting(float), required=False)

    """Crosslinker stub mod masses (modifications associated with crosslinker cleavage)"""
    cleavage_stubs = ListSetting(Stub, [], required=False)

    def __init__(self, **kwargs):
        """
        Initialise the Crosslinker.

        Forwards all kwargs to super().__init__ but preprocesses the specificity to parse out the
         termini information and whether it is homo- or heterobifunctional
        """
        super().__init__(**kwargs)

        if len(self.specificity) not in [1, 2]:
            message = "only crosslinker with one or two sets of specificities are supported"
            raise ValueError(message)

        self.nterm = []
        self.nterm_bonus = []
        self.cterm = []
        self.cterm_bonus = []
        self.ord_aa_specificity = []
        self.mod_specificity = []

        self.cleavage_stub_to_mass = {s.name: s.mass for s in self.cleavage_stubs}

        # check that specificity bonus is configured correctly
        if 'specificity_bonus' in self._values:
            if len(self.specificity) != len(self.specificity_bonus) or \
                    any([len(s[0]) != len(s[1])
                         for s in zip(self.specificity, self.specificity_bonus)]):
                raise ValueError("If specificity_bonus is defined its structure must match "
                                 "specificity!")
            # ToDo: remove this once heterobifunctional bonus is supported
            if len(self.specificity) == 2:
                raise AttributeError("specificity bonus for heterobifunctional crosslinkers is not"
                                     "supported yet!")
        else:
            # default fill with zeroes
            self.specificity_bonus = [[0] * len(n) for n in self.specificity]

        # check specificity for termini
        for i, sp in enumerate(self.specificity):
            if "nterm" in sp:
                self.nterm.append(True)
                nterm_index = sp.index('nterm')
                sp.remove("nterm")
                self.nterm_bonus.append(self.specificity_bonus[i][nterm_index])
                # remove the bonus from nterm
                del self.specificity_bonus[i][nterm_index]
            else:
                self.nterm.append(False)
                self.nterm_bonus.append(0)
            if "cterm" in sp:
                self.cterm.append(True)
                cterm_index = sp.index('cterm')
                sp.remove("cterm")
                self.cterm_bonus.append(self.specificity_bonus[i][cterm_index])
                # remove the bonus from cterm
                del self.specificity_bonus[i][cterm_index]
            else:
                self.cterm.append(False)
                self.cterm_bonus.append(0)

            # split the specificity up in aa and modification specificity
            if len(sp) > 0:
                aa_spec, mod_spec = zip(*((s[-1:], s[:-1]) for s in sp))
            else:
                aa_spec, mod_spec = ((), ())
            self.ord_aa_specificity.append([ord(aa) for aa in aa_spec])
            self.mod_specificity.append(mod_spec)

        self.homobifunctional = len(self.specificity) == 1
        self.cleavable = len(self.cleavage_stubs) > 0

    def hash(self):
        """
        Create a hash.

        :return: (int) hash
        """
        return hash((ConfigGroup.hash(self), tuple(self.nterm), tuple(self.cterm)))

    def to_dict(self, excl_defaults=True):
        values = super().to_dict(excl_defaults=excl_defaults)

        # restore the nterm and cterm information
        for i, n in enumerate(self.nterm):
            if n:
                values['specificity'][i].append("nterm")
                values['specificity_bonus'][i].append(self.nterm_bonus[i])

        for i, c in enumerate(self.cterm):
            if c:
                values['specificity'][i].append("cterm")
                values['specificity_bonus'][i].append(self.cterm_bonus[i])

        if values['specificity_bonus'] == [[0] * len(n) for n in values['specificity']]:
            del values['specificity_bonus']

        return values


Crosslinker.BS3 = Crosslinker(name='BS3',
                              mass=138.06807961,
                              specificity=[["K", "S", "T", "Y", "nterm"]],
                              specificity_bonus=[[0.2, 0, 0, 0, 0.2]])

Crosslinker.EDC = Crosslinker(name='EDC',
                              mass=-18.01056027,
                              specificity=[["K", "S", "T", "Y", "nterm"], ["E", "D", "cterm"]])

Crosslinker.SDA = Crosslinker(name='SDA',
                              mass=82.04186484,
                              specificity=[["K", "S", "T", "Y", "nterm"], ['X']],
                              cleavage_stubs=[
                                  Stub(name='0', mass=0, pairs_with=['S']),
                                  Stub(name='S', mass=82.04186484, pairs_with=['0']),
                              ])
Crosslinker.DSSO = Crosslinker(name='DSSO',
                               mass=158.0038,
                               specificity=[["K", "S", "T", "Y", "nterm"]],
                               specificity_bonus=[[0.2, 0, 0, 0, 0.2]],
                               cleavage_stubs=[
                                   Stub(name='a', mass=54.010565, pairs_with=['s', 't']),
                                   Stub(name='s', mass=103.993200, pairs_with=['a']),
                                   Stub(name='t', mass=85.982635, pairs_with=['a']),
                               ])


class Modification(ConfigGroup):
    """Modification configuration."""

    def __init__(self, **kwargs):
        """Initialise the Modification."""
        if 'mass' not in kwargs and 'composition' in kwargs:
            kwargs['mass'] = cmass.calculate_mass(formula=kwargs['composition'])
        super().__init__(**kwargs)

        self.ord_aa_specificity = [ord(aa) for aa in self.specificity]

        self.nterm_mod = self.name.endswith('-')
        self.cterm_mod = self.name.startswith('-')

        if self.long_name is None:
            self.long_name = self.name
    """
    Name of the modification, modX syntax:
        xx for side chain modification
        xx- for n-terminal modification
        -xx for c-terminal modification
    """
    name = Setting(str)

    """Long name (not actually used anywhere in xiSEARCH, only in xiADMIN & xiVIEW)"""
    long_name = Setting(str, default=None)

    """
    Level that the modification is applied on, i.e. if modifications happens post- or pre-digestion.
    Valid values: protein, peptide
    """
    level = Setting(str, default='protein', valid_values=('protein', 'peptide'))

    """
    amino acids that can be modified as a list of strings. Syntax:
        - amino acids in one letter code, e.g. "K"
        - "X" for any amino acid
        - "nterm" or "cterm" to specify SIDE-CHAIN modification of a terminal aa, e.g. "ntermA"
    """
    specificity = ListSetting(str)

    """chemical composition of the modification as a string, e.g. C2H3O1N1"""
    composition = Setting(str, required=False)

    """
    type of modification:
        - variable: both modified and unmodified versions exist
        - fixed: modification that is applied to every suitable amino acid
        - linear: only applied on linear peptides
        - known: from FASTA
        - ms3stub: stub from cleavable crosslinker in MS3 spectra (each peptide can have at most
                    one ms3stub modification independent of other settings)
    """
    type = Setting(str, valid_values=('variable', 'fixed', 'linear', 'known', 'ms3stub'))

    """mass in dalton"""
    mass = Setting(float, required=False)


class ModificationConfig(ConfigGroup):
    """Modification related configs."""

    modifications = ListSetting(Modification, [])

    """maximum number of variable modifications that happen before digestion"""
    max_var_protein_mods = Setting(int, 2)

    """maximum number of variable modifications that happen after digestion"""
    max_var_peptide_mods = Setting(int, 2)

    """maximum number of variable modifications that happen before digestion and are only
    considered for linear peptide matches"""
    max_linear_protein_mods = Setting(int, 2)

    """maximum number of variable modifications that happen after digestion and are only
    considered for linear peptide matches"""
    max_linear_peptide_mods = Setting(int, 2)

    """maximum number of modified variants to generate per peptide"""
    max_modified_peps = Setting(int, 20)

    @memoized_property
    def compositions(self):
        """
        Transform the chemical composition of modifications and amino acids into pyteomics syntax.

        :return: (dict) atomic compositions of modifications in pyteomics.mass.aa_comp syntax.
        """
        mod_compositions = {m.name: cmass.CComposition(m.composition) for m in self.modifications}
        aa_compositions = dict(cmass.std_aa_comp)
        aa_compositions.update(mod_compositions)

        return aa_compositions


class IsotopeDetectorConfig(ConfigGroup):
    """Isotope configuration."""

    """
    Relative tolerance for matching isotope clusters (defaults to ms2_rtol).
    This rtol is assumed for pairwise peak comparison on both peaks. Currently approximated by
    2 * rtol on the first peak.
    """
    rtol = Setting(float, -1)

    """
    Maximum Number of isotope peaks to look for in the initial state.
    Longer cluster will still be recognised - but in two steps.
    """
    cluster_calc_size = Setting(int, 7)

    """Only assume the start of a cluster if the ratio of the first peak to the second does not
    exceed this ratio. Stems mainly from observations with labeled peptides/crosslinker."""
    max_mono_to_first_peak_ratio = 8

    """Only try to break up clusters that are longer than this"""
    avergine_min_cluster_size = Setting(int, 5)

    """When the intensity of a peak is off by this factor assume a new cluster starts"""
    avergine_breakup_factor = Setting(int, 5)


class Enzyme(ConfigGroup):
    """Enzyme configuration."""

    """Name of the enzyme"""
    name = Setting(str)

    """regular expression rule for cleavage"""
    rule = Setting(str, required=False)

    """enzyme cleaves n-terminal of these amino-acids"""
    nterminal_of = ListSetting(str, [])

    """enzyme cleaves c-terminal of these amino-acids"""
    cterminal_of = ListSetting(str, [])

    """enzyme does not cleave if the opposing amino-acid is on of these"""
    restraining = ListSetting(str, [])

    def __init__(self, **kwargs):
        """Turn cterminal_of and nterminal_of into a regex rule."""
        super().__init__(**kwargs)
        if 'rule' in self._values:
            if len(self.nterminal_of) > 0 or len(self.cterminal_of) > 0:
                raise ValueError("Can't handle definition of a rule and separate definitions of "
                                 "digested amino-acids")
            return

        # cterminal
        if len(self.cterminal_of) > 0:
            # in the middle of the protein
            # # (?<=[A-Z]) prevents that we accept modified versions of the digested amino-acids
            # as digestable
            cterm_rule = '(?<=[A-Z])(' + '|'.join(self.cterminal_of) + ')'
            if len(self.restraining) > 0:
                cterm_rule += '(?!' + '|'.join(self.restraining) + ')'

            # at the beginning of a protein
            # we need this as a separate rule as here the "(?<=[A-Z])" condition will fail
            cterm_rule += '|(?<=^)(' + '|'.join(self.cterminal_of) + ')'
            if len(self.restraining) > 0:
                cterm_rule += '(?!' + '|'.join(self.restraining) + ')'

            # either of the two previous rules
            cterm_rule = '(' + cterm_rule + ')'

            # if we have no n-terminal condition, then this is the complete rule
            if len(self.nterminal_of) == 0:
                self._values['rule'] = cterm_rule
                return

        if len(self.nterminal_of) > 0:

            # any capital letter followed by the defined amino-acid
            # (?<=[A-Z]) means we don't have a modified amino-acid
            nterm_rule = '(?<=[A-Z])(?=' + '|'.join(self.nterminal_of) + ')'

            # don't digest if the digested amino-acids are after one of the restraining ones
            if len(self.restraining):
                nterm_rule = '(?<!' + '|'.join(self.restraining) + ')'

            if len(self.cterminal_of) == 0:
                self._values['rule'] = nterm_rule
                return

        if len(self.restraining) > 0:
            raise ValueError("currently we don't support having both c- and "
                             "n-terminal rules and a restraining conditions!")

        self._values['rule'] = "(" + cterm_rule + "|" + nterm_rule + ")"

    def to_dict(self, excl_defaults=True):
        if len(self.nterminal_of) > 0 or len(self.cterminal_of) > 0:
            return {
                key: self._values[key] for key in ['name', 'nterminal_of',
                                                   'cterminal_of', 'restraining']
            }
        else:
            return {
                key: self._values[key] for key in ['name', 'rule']
            }


# Simple Trypsin not cleaving modified Lysine or Arginine
Enzyme.trypsinRE = Enzyme(name='trypsin',
                          rule='((?<=[A-Z])[KR]|(?<=^)[KR])(?=[^A-Z]*[A-OQ-Z])')

Enzyme.trypsin = Enzyme(name='trypsin',
                        cterminal_of=['K', 'R'], restraining=['P'])

# simple asp-n not cleaving modified Aspartic Acid
Enzyme.asp_n = Enzyme(name='asp-n',
                      nterminal_of=["D"])

Enzyme.asp_nRE = Enzyme(name='asp-n',
                        rule='[A-Z](?=D)')


class DigestionConfig(ConfigGroup):
    """Digestion configuration."""

    def __init__(self, **kwargs):
        """Initialise the DigestionConfig."""
        super().__init__(**kwargs)
        if self.max_peptide_length > 252:
            raise ValueError("Peptides with more than 252 amino acids are not supported!")

    """Digestion mode ('parallel' or 'sequential')"""
    mode = Setting(str, 'parallel', valid_values=('parallel', 'sequential'))

    """Digestion enzymes in use"""
    enzymes = ListSetting(Enzyme, [Enzyme.trypsin])

    """Number of missed cleavages permitted"""
    missed_cleavages = Setting(int, 0)

    # following options are applied during digestion
    """Minimum peptide length filter."""
    min_peptide_length = Setting(int, 4)

    # Offer two options for control over the filter of heavy/long peptides.
    # Note that the approx_max_mass cut-off is translated into peptide length
    # by using the average mass of an amino acid. The filtering is then done on
    # the lower of the cut-offs.
    """Maximum peptide length filter."""
    max_peptide_length = Setting(int, 40)

    """Maximum approx. peptide mass filter (using avg. amino acid mass)."""
    approx_max_mass = Setting(int, 5000)


class Loss(ConfigGroup):
    """Neutral Loss configuration."""

    def __init__(self, **kwargs):
        """Initialise the Loss."""
        if 'mass' not in kwargs and 'composition' in kwargs:
            kwargs['mass'] = cmass.calculate_mass(formula=kwargs['composition'])

        super().__init__(**kwargs)
        specificity = self.specificity

        # check specificity for termini
        if "nterm" in specificity:
            self.nterm = True
            specificity.remove("nterm")
        else:
            self.nterm = False

        if "cterm" in specificity:
            self.cterm = True
            specificity.remove("cterm")
        else:
            self.cterm = False

        # split the specificity up in aa and modification specificity
        if len(specificity) > 0:
            aa_spec, mod_spec = zip(*((s[-1:], s[:-1]) for s in specificity))
        else:
            aa_spec, mod_spec = ((), ())
        self.ord_aa_specificity = [ord(aa) for aa in aa_spec]
        self.mod_specificity = mod_spec

    def to_dict(self, excl_defaults=True):
        values = super().to_dict(excl_defaults=excl_defaults)

        # restore the nterm and cterm information
        if self.nterm:
            values['specificity'].append("nterm")
        if self.cterm:
            values['specificity'].append("cterm")

        return values

    """Name of the loss"""
    name = Setting(str)

    """What can throw this loss. Amino acids in one letter code, "nterm" or "cterm" for termini"""
    specificity = ListSetting(str)

    """Chemical composition of the modification as a string, e.g. H201"""
    composition = Setting(str, required=False)

    """Mass loss in Dalton"""
    mass = Setting(float, required=False)


Loss.H2O = Loss(name='H2O', mass=18.01056027, specificity=['S', 'T', 'D', 'E', 'cterm'])
Loss.NH3 = Loss(name='NH3', mass=17.02654493, specificity=['R', 'K', 'N', 'Q', 'nterm'])


class FragmentationConfig(ConfigGroup):
    """Fragmentation configuration."""
    def __init__(self, **kwargs):
        """
        Initialise the Fragmentation.
        converts the ion-names into ascii
        """
        super().__init__(**kwargs)
        cterm_new = []
        for c in self.cterm_ions:
            cterm_new.append(c.encode('ascii'))
        nterm_new = []
        for n in self.nterm_ions:
            nterm_new.append(n.encode('ascii'))
        self.cterm_ions_ascii = cterm_new
        self.nterm_ions_ascii = nterm_new

    """N-terminal ions to search"""
    nterm_ions = ListSetting(str, ['b'], valid_values=('a', 'b', 'c'))

    """C-terminal ions to search"""
    cterm_ions = ListSetting(str, ['y'], valid_values=('x', 'y', 'z'))

    """Neutral losses to consider in search"""
    losses = ListSetting(Loss, [])

    """Maximum number of losses to consider"""
    max_nloss = Setting(int, 4)

    """Include precursor ions"""
    add_precursor = Setting(bool, True)

    """Match fragments also to the M+1 peak"""
    match_missing_monoisotopic = Setting(bool, True)

    """Maximal charge state for linear fragments"""
    max_linear_fragment_charge = Setting(int, -1)

    """Minimal charge state for crosslinked fragments"""
    min_crosslinked_fragment_charge = Setting(int, 0)

    def to_dict(self, excl_defaults=True):
        """
        Convert the FragmentationConfig to a dictionary.

        We need specific handling of the 'nterm_ions' and 'cterm_ions',
        as these need to be present downstream even if they are just
        the default values.

        :param excl_defaults: (bool) exclude default values
        :return: (dict) dictionary representation of the FragmentationConfig
        """
        conf_dict = super().to_dict(excl_defaults) or {}
        if 'nterm_ions' not in conf_dict:
            conf_dict['nterm_ions'] = [n.decode('ascii') for n in self.nterm_ions_ascii]
        if 'cterm_ions' not in conf_dict:
            conf_dict['cterm_ions'] = [c.decode('ascii') for c in self.cterm_ions_ascii]
        return conf_dict


class FastaReaderConfig(ConfigGroup):
    """FastaReader configuration."""

    """Regular expression used for matching protein name"""
    re_name = Setting(str, "(?:sp|tr)\\|(?:[\\w-]+)\\|([^\\s]*)")

    """Regular expression used for matching protein description"""
    re_desc = Setting(str, "(?:sp|tr)\\|(?:[\\w-]+)\\|(.*)")

    """Regular expression used for matching protein accession"""
    re_accession = Setting(str, "(?:sp|tr)\\|([\\w-]+)\\|.*")


class DecoyConfig(ConfigGroup):
    """Decoy database configuration."""

    """How to generate decoys"""
    mode = Setting(str, 'reverse', valid_values=['random', 'reverse', 'shuffle', 'none'])

    """Prefix to use for decoy proteins"""
    prefix = Setting(str, 'REV_')

    """Should the decoy generation be aware of enzyme specificities?"""
    enzyme_aware = Setting(bool, True)

    """The actual method used for generating the decoys."""
    # TODO change that to an actual type definition
    protein_decoy_function = Setting(type(lambda x: x), required=False)


class DenoiseConfig(ConfigGroup):
    """Denoise Filter configuration."""

    """ Top N (intensity) peaks to return when denoising, per given interval """
    top_n = Setting(int)

    """ Interval in m/z to use when denoising for which to return the top N peaks """
    bin_size = Setting(int)


class ReportingRequirementsConfig(ConfigGroup):
    """
    Define settings that affect what results are written out.

    All matches (incl. top-ranking) are filtered to have at least `minimum_observed_fragment_sites`.

    Unless report_top_ranking_only is set at least the top two ranking matches are reported.
    Beyond that, `minimum_spectrum_coverage` and `delta_score_filter` define what gets reported.
    """

    """Filter results to top-ranking only before writing out"""
    report_top_ranking_only = Setting(bool, True)

    """Filter results to have at least this number of unique primary fragmentation-sites observed"""
    minimum_observed_fragment_sites = Setting(int, 2)

    """
    Top two matches get reported and beyond that only matches that explain at least this amount
    of the spectrum. This applies to both the spectrum intensity and the number of peaks.
    """
    minimum_spectrum_coverage = Setting(float, 0.05)

    """
    Top two matches get reported and beyond that only matches where the
    match_score is larger then the difference to the second ranked match
    """
    delta_score_filter = Setting(bool, True)


class ResultWriterConfig(ConfigGroup):
    """
    Define settings that affect how results are written out.
    """

    """Batch size for writing out results."""
    pg_batch_size = Setting(int, 100)


class RecalibrationConfig(ConfigGroup, ToleranceContainer):
    """
    Define mass recalibration.

    The file specific MS2 tolerances are used as isotope tolerances if they are smaller than the
    isotope tolerance set in the config.
    """
    def __init__(self, **kwargs):
        """
        Initialise the RecalibrationConfig.

        Forwards all kwargs to super().__init__ and translates MS tolerances.
        """
        super().__init__(**kwargs)

        # translate the tolerances from string to numeric atol/rtol
        self.initialise_tolerances()

        # translate errors to numeric values
        self.ms1_aerror, self.ms1_rerror = self.translate_ms_tol(self.ms1_error)
        self.ms2_aerror, self.ms2_rerror = self.translate_ms_tol(self.ms2_error)

    file = Setting(str)
    ms1_error = Setting(str, '0 ppm', valid_values=ToleranceContainer._re_ms_tol)
    ms1_tol = Setting(str, '0 ppm', valid_values=ToleranceContainer._re_ms_tol)
    ms2_error = Setting(str, '0 ppm', valid_values=ToleranceContainer._re_ms_tol)
    ms2_tol = Setting(str, '0 ppm', valid_values=ToleranceContainer._re_ms_tol)


class Config(ConfigGroup, ToleranceContainer):
    """Top level configuration for a search."""

    def __init__(self, **kwargs):
        """
        Initialise the Config.

        Forwards all kwargs to super().__init__ and translates loss specificity and crosslinker
        specificities. Also does some validity checks.
        """
        super().__init__(**kwargs)

        # translate the tolerances from string to numeric atol/rtol
        self.initialise_tolerances()

        # translate the modification specificities into indexes for all losses
        mod_names = [''] + [mod.name for mod in self.modification.modifications]
        for loss in self.fragmentation.losses:
            loss.mod_specificity_codes = [
                mod_names.index(mod) for mod in loss.mod_specificity if mod in mod_names]

        # ... and for all cross-linker
        if "crosslinker" in self._values:
            for xl in self.crosslinker:
                xl.mod_specificity_codes = [[mod_names.index(mod) for mod in end]
                                            for end in xl.mod_specificity]

        # if isotope tolerance is not explicitly set use the ms2 relative tolerance
        if self.isotope_config.rtol < 0:
            self.isotope_config.rtol = self.ms2_rtol

        # check max_alpha value
        if self.max_alpha_candidates < self.top_n_alpha_scores:
            raise ValueError("max_alpha_candidates should not be smaller than top_n_alpha_scores!")

    def to_dict(self, excl_defaults=True):
        values = super().to_dict(excl_defaults=excl_defaults)

        if excl_defaults and self.isotope_config.rtol == self.ms2_rtol:
            del values['isotope_config']['rtol']

        return values

    """
    Max number of threads to use for processing spectra. Setting to 0 means using the
    multiprocessing default, which is the value of `cpu_count`. Setting it to a negative
    number N means use all but minus N threads.
    """
    threads = Setting(int, 0)

    """Tolerance for matching precursor (MS1) m/z values."""
    ms1_tol = Setting(str, '3 ppm', valid_values=ToleranceContainer._re_ms_tol)

    """Tolerance for matching fragment (MS2) m/z values."""
    ms2_tol = Setting(str, '10 ppm', valid_values=ToleranceContainer._re_ms_tol)

    """
    Isotope error to consider for matching beta candidates according to xiMPA.
    Given any number greater than 0 beta candidates will not only be extracted
    for exactly matching precursor masses but *also* for masses that are
    (approximately) -1, -2 *and* -n Da lighter than the reported mass of the precursor
    to account for missing (lighter) isotope peaks.
    Examples:
        0 - (default) only looks for the given m/z from the MS2 spectrum
        2 -  will look for the given m/z and 2 missing (lighter) isotope peak.
    Recommendations for acquisitions > 2018:
        Thermo instruments: 2 (faster) / 3 (more sensitive)
    """
    isotope_error_ximpa = Setting(int, 0)

    """Crosslinker in use"""
    crosslinker = ListSetting(Crosslinker, [])

    """Fasta reader config"""
    fasta = Setting(FastaReaderConfig, FastaReaderConfig())

    """Isotope processor config"""
    isotope_config = Setting(IsotopeDetectorConfig, IsotopeDetectorConfig())

    """Digestion config"""
    digestion = Setting(DigestionConfig, DigestionConfig())

    """Fragmentation config"""
    fragmentation = Setting(FragmentationConfig, FragmentationConfig())

    """Number of top unique alpha candidate scores to consider"""
    top_n_alpha_scores = Setting(int, 10, max_value=10000)

    """Hard cap of top alpha candidates to consider"""
    max_alpha_candidates = Setting(int, 1000)

    """Number of top unique alpha beta candidate scores to consider"""
    top_n_alpha_beta_scores = Setting(int, 10, max_value=10000)

    """Modification configs"""
    modification = Setting(ModificationConfig, ModificationConfig())

    """Regular expression used for matching the scan number"""
    re_scan_number = Setting(str, "(?:scan=|[^.]*\\.)([0-9]+)(?:\\.\1)?")

    """Regular expression used for matching the run name"""
    re_run_name = Setting(str, "^([^\\s.]+)")

    """Linearise a spectrum based on charge or on mass"""
    linearisation = Setting(str, "charge", valid_values=("mass", "charge"))

    """ Denoise settings for the alpha candidate selection """
    denoise_alpha = Setting(DenoiseConfig, DenoiseConfig(top_n=10, bin_size=100))

    """ Denoise settings for the alpha_beta scoring """
    denoise_alpha_beta = Setting(DenoiseConfig, DenoiseConfig(top_n=20, bin_size=100))

    """Number of neutral loss fragments needed to fulfill the multi-loss requirement
    for conservative scoring"""
    conservative_n_multi_loss = Setting(int, 3)

    """Define how to generate decoys"""
    decoy = Setting(DecoyConfig, DecoyConfig())

    """Filter applied before writing out"""
    reporting_requirements = Setting(ReportingRequirementsConfig, ReportingRequirementsConfig())

    """Search non-covalently bound peptides"""
    noncovalent_peptides = Setting(bool, False)

    """Syntax to use for writing modified peptide sequences out."""
    mod_peptide_syntax = Setting(str, 'modX', valid_values=('modX', 'Xmod'))

    """Define some settings affecting how results are written out."""
    result_writer = Setting(ResultWriterConfig, ResultWriterConfig())

    """Define recalibration errors."""
    recalibration = ListSetting(RecalibrationConfig, [])

    """ Should recalibration be applied to the MS1 and MS2 spectra? """
    recalibrate_spectra = Setting(bool, False)


class ConfigReader:
    """Config Reader class."""

    @classmethod
    def load_file(cls, file_name):
        """Open a file by filename and create a Config from it."""
        with open(file_name) as f:
            if file_name.lower().endswith('.json'):
                return cls.load_json(f)
            elif file_name.lower().endswith('.yaml') or file_name.lower().endswith('.yml'):
                return cls.load_yaml(f)
            else:
                # Guess format
                try:
                    return cls.load_json(f)
                except Exception:
                    return cls.load_yaml(f)

    @classmethod
    def load_json(cls, file_obj):
        """Create a Config from a JSON file."""
        settings = json.load(file_obj)
        config = Config(**settings)
        return config

    @classmethod
    def load_yaml(cls, file_obj):
        """Create a Config from a YAML file."""
        settings = yaml.safe_load(file_obj)
        config = Config(**settings)
        return config

    @classmethod
    def loads_json(cls, s):
        """Create a Config from a JSON string."""
        settings = json.loads(s)
        config = Config(**settings)
        return config

    @classmethod
    def loads_yaml(cls, s):
        """Create a Config from a YAML string."""
        settings = yaml.safe_load(s)
        config = Config(**settings)
        return config
