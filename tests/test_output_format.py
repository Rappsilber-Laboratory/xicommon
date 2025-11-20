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

from xicommon.mock_context import MockContext
from xicommon.output_format import *
from xicommon.config import Config, ModificationConfig

mods = [
    {'name': 'm1', 'specificity': ['A'], 'type': 'variable', 'composition': 'H1'},
    {'name': 'm2', 'specificity': ['L'], 'type': 'variable', 'composition': 'H2'},
    {'name': 'n-', 'specificity': ['X'], 'type': 'variable', 'composition': 'H3'},
    {'name': '-c', 'specificity': ['X'], 'type': 'variable', 'composition': 'H4'},
]
mod_cfg = ModificationConfig(modifications=mods)
cfg = Config(modification=mod_cfg)
ctx = MockContext(cfg)

peptides = np.array([b'ALA', b'm1Am2Lm1A', b'n-m1Am2Lm1A-c'])
ctx.setup_peptide_db(peptides)


def test_create_mod_str():
    mod_strs = create_mod_str(ctx, [0, 1, 2])
    assert mod_strs == ['', 'm1;m2;m1', 'n-;m1;m2;m1;-c']


def test_create_mod_pos_str():
    mod_pos_strs = create_mod_pos_str(ctx, [0, 1, 2])
    assert mod_pos_strs == ['', '1;2;3', 'nterm;1;2;3;cterm']


def test_create_mod_mass_str():
    mod_mass_strs = create_mod_mass_str(ctx, [0, 1, 2])

    assert mod_mass_strs == [
        '',
        '1.00782503207;2.01565006414;1.00782503207',
        '3.02347509621;1.00782503207;2.01565006414;1.00782503207;4.03130012828'
    ]
