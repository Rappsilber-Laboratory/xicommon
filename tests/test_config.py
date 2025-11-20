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

import tempfile

import yaml

from xicommon.config import Config, Setting, ConfigGroup, ListSetting, Loss, Stub, \
    Crosslinker, ConfigReader, ModificationConfig, Modification, Enzyme, DigestionConfig
from numpy.testing import assert_almost_equal
import io
import json
import pytest
import re
import os


def test_config_system():
    """ Test the configuration system """

    class SubConfig(ConfigGroup):
        v1 = Setting(int, 1)
        v2 = Setting(str, "hello")
        v3 = ListSetting(int, [1, "2"])

    class TestConfig(ConfigGroup):
        i1 = Setting(int, 1)
        i2 = Setting(int, '2')
        f1 = Setting(float, 3.0)
        f2 = Setting(float, 4)
        s1 = Setting(str, "five")
        s2 = Setting(str, 5)
        d1 = Setting(int, 1)
        d2 = Setting(int, 2)
        l1 = ListSetting(int, [1, "2"])
        l2 = ListSetting(str, [1, "2", "three"])
        sub1 = Setting(SubConfig)
        sub2 = Setting(SubConfig, SubConfig(v1=2, v2="bye"))
        v_list = Setting(int, valid_values=[1, 2, 3], default=2)
        v_re = Setting(str, valid_values=re.compile('[A-z]+$'), required=False)
        v_required = Setting(str, required=True)
        i_max = Setting(int, 5, max_value=10)
        f_max = Setting(float, 2.0, max_value=3.5)

    assert issubclass(TestConfig, ConfigGroup)

    config = TestConfig(d1=4, sub1=SubConfig(), v_required='str')

    assert isinstance(config, TestConfig)

    expected_results = dict(i1=1, i2=2, f1=3.0, f2=4.0, s1="five", s2="5", d1=4, d2=2, l1=[1, 2],
                            l2=['1', '2', 'three'], i_max=5, f_max=2.0)

    for setting, expected_value in expected_results.items():
        value = getattr(config, setting)
        assert type(value) is type(expected_value)
        assert value == expected_value

    assert config.sub1.v1 == 1
    assert config.sub1.v2 == "hello"
    assert config.sub2.v1 == 2
    assert config.sub2.v2 == "bye"
    assert config.sub2.v3 == [1, 2]

    # missing required attribute
    with pytest.raises(AttributeError):
        TestConfig(d1=4, sub1=SubConfig())

    with pytest.raises(AttributeError):
        config.nonexistent

    with pytest.raises(KeyError):
        TestConfig(nonexistent=1)

    with pytest.raises(TypeError):
        Setting(int, 'three')

    with pytest.raises(TypeError):
        ListSetting(int, ['three', 2])

    with pytest.raises(ValueError):
        TestConfig(v_list=4)

    with pytest.raises(TypeError):
        TestConfig(v_list="four")

    with pytest.raises(ValueError):
        TestConfig(v_re="Ab3")

    with pytest.raises(ValueError):
        Setting(int, 3, valid_values=[1, 2])

    with pytest.raises(ValueError):
        Setting(int, 11, max_value=10)

    with pytest.raises(ValueError):
        Setting(int, 10.1, max_value=10)

    with pytest.raises(ValueError):
        Setting(float, 5.6, max_value=5.5)

    with pytest.raises(ValueError):
        Setting(float, 6, max_value=5.9)

    # max_value not supported for str type
    with pytest.raises(TypeError):
        Setting(str, 1.0, max_value=5.5)


def test_crosslinker_hash():

    # Make sure that the hashing correctly distinguishes nterm/cterm differences.
    xl1 = Crosslinker(name='test', mass=100, specificity=[["K", "S", "nterm"]])
    xl2 = Crosslinker(name='test', mass=100, specificity=[["K", "S", "nterm"]])
    xl3 = Crosslinker(name='test', mass=100, specificity=[["K", "S", "cterm"]])

    assert xl1.hash() == xl2.hash()
    assert xl2.hash() != xl3.hash()


def test_modification_can_be_build_through_modification_config():
    values = [{
        'name': 'ox',
        'specificity': ['M'],
        'type': 'variable',
        'composition': 'O1'
    }]
    cfg = ModificationConfig(modifications=values).modifications[0]
    assert cfg.name == 'ox'
    assert cfg.specificity == ['M']
    assert cfg.type == 'variable'
    assert cfg.composition == 'O1'
    assert_almost_equal(cfg.mass, 15.99491461956)


def test_modification_config_accepts_modification_objects_or_dicts():
    values = [{'name': 'ox', 'specificity': ['M'], 'type': 'variable'},
              Modification(name='cm', specificity=['C'], type='fixed')]
    cfg = ModificationConfig(modifications=values).modifications
    assert cfg[0].name == 'ox'
    assert cfg[1].name == 'cm'


def json_io_from_dict(dict):
    configfile = io.StringIO()
    configfile.write(
        json.dumps(dict)
    )
    configfile.seek(0)
    return configfile


def yaml_io_from_dict(dict):
    configfile = io.StringIO()
    configfile.write(
        yaml.dump(dict)
    )
    configfile.seek(0)
    return configfile


def test_crosslinker_config():
    with pytest.raises(AttributeError):
        Crosslinker()

    with pytest.raises(AttributeError):
        Crosslinker(name="test")

    with pytest.raises(AttributeError):
        Crosslinker(name="test", mass=123)

    with pytest.raises(AttributeError):
        Crosslinker(mass=123, specificity=['K', 'S', 'T', 'Y'])

    with pytest.raises(AttributeError):
        Crosslinker(name="test", specificity=['K', 'S', 'T', 'Y'])

    with pytest.raises(AttributeError):
        Crosslinker(name="test", mass=123)

    with pytest.raises(ValueError):
        Crosslinker(name="test", mass=123,
                    specificity=['K', 'S', 'T', 'Y'], specificity_bonus=[0, 0, 0])

    with pytest.raises(ValueError):
        Crosslinker(name="test", mass=123,
                    specificity=[['K', 'S', 'T', 'Y'], ['X']], specificity_bonus=[0, 0, 0])

    # should fail because of more then two specificity lists
    with pytest.raises(ValueError):
        Crosslinker(mass=123, specificity=[['K'], ['S'], ['T'], ['Y']], name="test")

    test_cl = Crosslinker(mass=123, specificity=['K', 'S', 'T', 'Y'], name="test")
    assert test_cl.specificity_bonus == [[0, 0, 0, 0]]
    assert test_cl.cterm_bonus == [0]
    assert test_cl.cterm == [False]
    assert test_cl.nterm_bonus == [0]
    assert test_cl.nterm == [False]

    # ToDo: change this once heterobifunctional bonus is supported
    with pytest.raises(AttributeError):
        test_cl = Crosslinker(name="test", mass=123,
                              specificity=[['K', 'S', 'T', 'Y', 'cterm'], ['X']],
                              specificity_bonus=[[0, 0, 0.5, 0, 0.1], [0]])
    # assert test_cl.specificity_bonus == [[0, 0, 0.5, 0], [0]]
    # assert test_cl.cterm_bonus == [0.1, 0]
    # assert test_cl.cterm == [True, False]
    # assert test_cl.nterm_bonus == [0, 0]
    # assert test_cl.nterm == [False, False]

    test_cl = Crosslinker(name="test", mass=123, specificity=['K', 'S', 'T', 'Y', 'nterm'],
                          specificity_bonus=[0.2, 0, 0.05, 0, 0.1])
    assert test_cl.specificity_bonus == [[0.2, 0, 0.05, 0]]
    assert test_cl.nterm_bonus == [0.1]
    assert test_cl.nterm == [True]
    assert test_cl.cterm_bonus == [0]
    assert test_cl.cterm == [False]


def test_list_setting_valid_values():
    class ListTestConfig(ConfigGroup):
        lst = ListSetting(int, [1], valid_values=[1, 2, 3])
        list_of_lists = ListSetting(ListSetting(int, [1], valid_values=[1, 2, 3]), required=False)

    # Basic case of a ListSetting
    assert ListTestConfig(lst=[1, 2, 3]).lst == [1, 2, 3]
    assert ListTestConfig(lst=[2, 3]).lst == [2, 3]
    assert ListTestConfig(lst=["1", 2, 3]).lst == [1, 2, 3]
    assert ListTestConfig(lst=1).lst == [1]
    # test if non valid values are cought
    with pytest.raises(ValueError) as _:
        ListTestConfig(lst=[1, 2, 3, 4])
    with pytest.raises(ValueError) as _:
        ListTestConfig(lst=[4])

    # Nested ListSettings
    assert ListTestConfig(list_of_lists=[[1, 2, 3]]).list_of_lists == [[1, 2, 3]]
    assert ListTestConfig(list_of_lists=[["1", 2, 3]]).list_of_lists == [[1, 2, 3]]
    assert ListTestConfig(list_of_lists=[["1", 2], [3]]).list_of_lists == [[1, 2], [3]]
    assert ListTestConfig(list_of_lists=["1", 2, 3]).list_of_lists == [[1, 2, 3]]
    # again 4 is not a valid value and should throw an error
    with pytest.raises(ValueError) as _:
        ListTestConfig(list_of_lists=[["4", 2], [3]]).list_of_lists


def test_modifying_config():

    config1 = Config(modification=ModificationConfig(max_var_protein_mods=2))
    config2 = Config(modification=ModificationConfig(max_var_protein_mods=3))

    assert config1.modification.max_var_protein_mods == 2
    assert config2.modification.max_var_protein_mods == 3

    hash1 = config1.modification.hash()
    hash2 = config2.modification.hash()

    assert hash1 != hash2

    config1.modification.max_var_protein_mods = 3

    assert config1.modification.max_var_protein_mods == 3

    hash3 = config1.modification.hash()

    assert hash3 == hash2


class TestConfigReader():
    def test_reader_returns_a_config(self):
        f = json_io_from_dict({})
        assert type(ConfigReader.load_json(f)) == Config

    def test_reader_accept_an_io_as_arg(self):
        f = json_io_from_dict({})
        assert ConfigReader.load_json(f)

    def test_reader_loads_simple_values(self):
        f = json_io_from_dict({'isotope_error_ximpa': 3})
        config = ConfigReader.load_json(f)
        assert config.isotope_error_ximpa == 3

    def test_reader_maintains_defaults(self):
        f = json_io_from_dict({})
        config = ConfigReader.load_json(f)
        assert config.isotope_error_ximpa == 0
        assert config.top_n_alpha_scores == 10

    def test_reader_loads_nested_values(self):
        f = json_io_from_dict({
            'ms2_tol': '1 Da',
            'crosslinker': [{
                'name': 'tst',
                'mass': 200,
                'specificity': [['K']]
            }]
        })
        config = ConfigReader.load_json(f)
        assert config.ms2_atol == 1
        assert config.crosslinker[0].mass == 200
        assert config.crosslinker[0].specificity == [['K']]

    def test_reader_loads_setting_constants(self):
        f = json_io_from_dict({
            'crosslinker': ['BS3']
        })
        config = ConfigReader.load_json(f)
        assert config.crosslinker[0].mass == 138.06807961
        assert config.crosslinker[0].name == 'BS3'
        assert config.crosslinker[0].specificity == [['K', 'S', 'T', 'Y']]
        assert config.crosslinker[0].nterm == [True]

    def test_reader_loads_list_setting_constants(self):
        f = json_io_from_dict({
            'digestion': {
                'enzymes': ['trypsin', 'asp_n']
            }
        })
        config = ConfigReader.load_json(f)
        e1, e2 = config.digestion.enzymes
        assert e1.name == 'trypsin'
        assert e1.rule == Enzyme.trypsin.rule
        assert e2.name == 'asp-n'
        assert e2.rule == Enzyme.asp_n.rule

    def test_reader_loads_nested_values_in_array(self):
        f = json_io_from_dict({
            'modification': {
                'modifications': [
                    {
                        'name': 'cm',
                        'specificity': ['C'],
                        'type': 'fixed',
                        'composition': 'C2H3N1O1'
                    },
                    {
                        'name': 'ox',
                        'specificity': ['M'],
                        'type': 'variable',
                        'composition': 'O1'
                    }
                ]}
        })
        config = ConfigReader.load_json(f)
        mod1, mod2 = config.modification.modifications
        assert mod1.name == 'cm'
        assert mod1.type == 'fixed'
        assert mod1.composition == 'C2H3N1O1'
        assert mod1.specificity == ['C']
        assert_almost_equal(mod1.mass, 57.02146372057)
        assert mod2.name == 'ox'
        assert mod2.type == 'variable'
        assert mod2.composition == 'O1'
        assert mod2.specificity == ['M']
        assert_almost_equal(mod2.mass, 15.99491461956)

    def test_yaml_reader_loads_nested_values_in_array(self):
        f = yaml_io_from_dict({
            'modification': {
                'modifications': [
                    {
                        'name': 'cm',
                        'specificity': ['C'],
                        'type': 'fixed',
                        'composition': 'C2H3N1O1'
                    },
                    {
                        'name': 'ox',
                        'specificity': ['M'],
                        'type': 'variable',
                        'composition': 'O1'
                    }
                ]}
        })
        config = ConfigReader.load_yaml(f)
        mod1, mod2 = config.modification.modifications
        assert mod1.name == 'cm'
        assert mod1.type == 'fixed'
        assert mod1.composition == 'C2H3N1O1'
        assert mod1.specificity == ['C']
        assert_almost_equal(mod1.mass, 57.02146372057)
        assert mod2.name == 'ox'
        assert mod2.type == 'variable'
        assert mod2.composition == 'O1'
        assert mod2.specificity == ['M']
        assert_almost_equal(mod2.mass, 15.99491461956)

    def test_reader_loads_empty_nested_values_in_array(self):
        f = json_io_from_dict({
            'modification': {
                'modifications': [
                ]
            }
        })
        config = ConfigReader.load_json(f)
        assert config.modification.modifications == []

    def test_reader_loads_file(self):
        path = './tests/fixtures/test_run_config.json'
        config = ConfigReader.load_file(path)
        assert config.crosslinker[0].mass == 138.06807961
        assert len(config.digestion.enzymes) == 1
        assert config.digestion.enzymes[0].name == 'trypsin'
        assert config.digestion.enzymes[0].rule == Enzyme.trypsin.rule

        assert config.digestion.missed_cleavages == 0
        assert config.digestion.min_peptide_length == 3
        assert config.ms1_tol == '5ppm'
        assert_almost_equal(config.ms1_rtol, 5e-6, decimal=10)
        assert config.ms1_atol == 0
        assert config.ms2_tol == '15ppm'
        assert_almost_equal(config.ms2_rtol, 15e-6, decimal=10)
        assert config.ms2_atol == 0
        assert config.top_n_alpha_scores == 10

    def test_reader_loads_yaml_file(self):
        path = './tests/fixtures/test_run_config_mods.json'
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, 'config.yaml')
            with open(path) as f:
                conf_dict = json.load(f)
            with open(yaml_path, 'w') as f:
                yaml.dump(conf_dict, f)
            config = ConfigReader.load_file(yaml_path)
            config_json = ConfigReader.load_file(path)
            assert config == config_json
            assert config.crosslinker[0].mass == 138.06807961
            assert len(config.digestion.enzymes) == 1
            assert config.digestion.enzymes[0].name == 'trypsin'
            assert config.digestion.enzymes[0].rule == Enzyme.trypsin.rule

            assert config.digestion.missed_cleavages == 0
            assert config.digestion.min_peptide_length == 3
            assert config.ms1_tol == '5ppm'
            assert_almost_equal(config.ms1_rtol, 5e-6, decimal=10)
            assert config.ms1_atol == 0
            assert config.ms2_tol == '15ppm'
            assert_almost_equal(config.ms2_rtol, 15e-6, decimal=10)
            assert config.ms2_atol == 0
            assert config.top_n_alpha_scores == 10

    def test_reader_loads_file_with_mods(self):
        path = './tests/fixtures/test_run_config_mods.json'
        config = ConfigReader.load_file(path)
        assert len(config.modification.modifications) == 2
        mod1, mod2 = config.modification.modifications
        assert mod1.name == 'cm'
        assert mod1.specificity == ["C"]
        assert mod1.type == 'fixed'
        assert mod1.composition == "C2H3N1O1"
        assert mod2.name == 'ox'
        assert mod2.specificity == ["M"]
        assert mod2.type == 'variable'
        assert mod2.composition == "O1"


def test_enzyme_cnterm_restraining_fail():

    with pytest.raises(ValueError):
        Config(digestion=DigestionConfig(enzymes=Enzyme(name='not_trypsin',
                                         cterminal_of=['K', 'R'], nterminal_of=['K', 'R'],
                                         restraining=['P']), mode='sequential'))


def test_long_crosslinkername():
    # this should work
    Crosslinker(name='123456789012345678901234567890123456789', mass=100,
                     specificity=[["K"]])

    # this should fail
    with pytest.raises(ValueError):
        Crosslinker(name='1234567890123456789012345678901234567890', mass=100,
                    specificity=[["K"]])


def test_max_alpha_candidates():
    Config(max_alpha_candidates=1000)
    Config(max_alpha_candidates=10, top_n_alpha_scores=10)

    # this should fail - smaller max_alpha_candidates
    with pytest.raises(ValueError):
        Config(max_alpha_candidates=10, top_n_alpha_scores=20)

    # smaller max_alpha_candidates as default top_n_alpha (10)
    with pytest.raises(ValueError):
        Config(max_alpha_candidates=9)


def test_subconfig_decoupling():
    """Test that two configs are decoupled from each other."""
    config1 = Config()
    # default is enzyme aware True
    assert config1.decoy.enzyme_aware
    # set it to false
    config1.decoy.enzyme_aware = False
    assert not config1.decoy.enzyme_aware

    # create second config and check enzyme aware. Should still be default (True)
    config2 = Config()
    assert config2.decoy.enzyme_aware
    # config1 should still be False
    assert not config1.decoy.enzyme_aware

    # set config2 explicitly to True
    config2.decoy.enzyme_aware = True
    # config 2 should still be True and config1 should still be False
    assert config2.decoy.enzyme_aware
    assert not config1.decoy.enzyme_aware


def test_listsetting_single_value():
    class TestConfig(ConfigGroup):
        ls = ListSetting(Setting(str))
    tc = TestConfig(ls='test')
    assert tc.ls == ['test']


def test_listsetting_multiple_setting_values():
    class TestConfig(ConfigGroup):
        ls = ListSetting(Setting(str))
    tc = TestConfig(ls=['test1', 'test2'])
    assert tc.ls == ['test1', 'test2']


def test_enzyme_c_and_nterm_rule():
    ec = Enzyme(name='test', cterminal_of=['K', 'R'])
    en = Enzyme(name='test', nterminal_of=['K', 'R'])
    ecn = Enzyme(name='test', cterminal_of=['K', 'R'], nterminal_of=['K', 'R'])

    assert ecn.rule == f'({ec.rule}|{en.rule})'


def test_enzyme_term_and_rule():
    # defining cterminal_of/nterminal of as well as a rule should raise an error
    with pytest.raises(ValueError):
        Enzyme(name='test', cterminal_of=['K', 'R'], nterminal_of=['K', 'R'], rule='[KR]')


class TestConfigWriter():
    def test_config_writer(self, tmpdir):
        path = './tests/fixtures/test_run_config.json'
        config = ConfigReader.load_file(path)

        # write the config to a file (uses to_dict)
        out_path = os.path.join(tmpdir, "test_config.json")
        config.write(out_path, excl_defaults=False)

        config_reloaded = ConfigReader.load_file(out_path)

        # dicts are equal if they have the same key value pairs
        assert config.to_dict() == config_reloaded.to_dict()
        assert config == config_reloaded

    def test_config_to_dict(self):
        path = './tests/fixtures/test_run_config.json'
        config = ConfigReader.load_file(path)
        config_dict = config.to_dict(excl_defaults=False)
        assert len(config_dict['crosslinker']) == 1
        assert config_dict['crosslinker'][0]['mass'] == 138.06807961
        assert config_dict['crosslinker'][0]['specificity'] == [['K', 'S', 'T', 'Y', 'nterm']]
        assert config_dict['crosslinker'][0]['specificity_bonus'] == [[0.2, 0.0, 0.0, 0.0, 0.2]]
        assert len(config_dict['digestion']['enzymes']) == 1
        assert config_dict['digestion']['enzymes'][0]['name'] == 'trypsin'
        assert 'rule' not in config_dict['digestion']['enzymes'][0]
        assert config_dict['digestion']['enzymes'][0]['cterminal_of'] == ['K', 'R']
        assert config_dict['digestion']['enzymes'][0]['restraining'] == ['P']
        assert config_dict['digestion']['min_peptide_length'] == 3
        assert config_dict['digestion']['missed_cleavages'] == 0
        assert config_dict['ms1_tol'] == '5ppm'
        assert 'ms1_rtol' not in config_dict
        assert 'ms1_atol' not in config_dict
        assert config_dict['ms2_tol'] == '15ppm'
        assert 'ms2_rtol' not in config_dict
        assert 'ms2_atol' not in config_dict
        assert config_dict['top_n_alpha_scores'] == 10
        assert not config_dict['reporting_requirements']["report_top_ranking_only"]
        assert config_dict['reporting_requirements']["minimum_observed_fragment_sites"] == 0
        assert config_dict['reporting_requirements']["minimum_spectrum_coverage"] == 0
        assert not config_dict['reporting_requirements']["delta_score_filter"]
        assert_almost_equal(config_dict['isotope_config']['rtol'], 15e-6, decimal=10)

    def test_config_to_dict_excl_defaults(self):
        path = './tests/fixtures/test_run_config.json'
        config = ConfigReader.load_file(path)
        config_dict = config.to_dict(excl_defaults=True)
        assert len(config_dict['crosslinker']) == 1
        assert config_dict['crosslinker'][0]['mass'] == 138.06807961
        assert config_dict['crosslinker'][0]['specificity'] == [['K', 'S', 'T', 'Y', 'nterm']]
        assert config_dict['crosslinker'][0]['specificity_bonus'] == [[0.2, 0.0, 0.0, 0.0, 0.2]]
        assert len(config_dict['digestion']['enzymes']) == 1
        assert config_dict['digestion']['enzymes'][0]['name'] == 'trypsin'
        assert 'rule' not in config_dict['digestion']['enzymes'][0]
        assert config_dict['digestion']['enzymes'][0]['cterminal_of'] == ['K', 'R']
        assert config_dict['digestion']['enzymes'][0]['restraining'] == ['P']
        assert config_dict['digestion']['min_peptide_length'] == 3
        assert 'missed_cleavages' not in config_dict['digestion']
        assert config_dict['ms1_tol'] == '5ppm'
        assert 'ms1_rtol' not in config_dict
        assert 'ms1_atol' not in config_dict
        assert config_dict['ms2_tol'] == '15ppm'
        assert 'ms2_rtol' not in config_dict
        assert 'ms2_atol' not in config_dict
        assert 'top_n_alpha_scores' not in config_dict
        assert not config_dict['reporting_requirements']["report_top_ranking_only"]
        assert config_dict['reporting_requirements']["minimum_observed_fragment_sites"] == 0
        assert config_dict['reporting_requirements']["minimum_spectrum_coverage"] == 0
        assert not config_dict['reporting_requirements']["delta_score_filter"]
        assert 'rtol' not in config_dict['isotope_config']

    def test_config_to_dict_with_commandline_options(self, tmpdir):
        config = Config()

        # syntax from __main__.py
        for o in [
            "reporting_requirements.report_top_ranking_only=False"
        ]:
            exec("config." + o)

        out_path = os.path.join(tmpdir, "test_config.json")
        config.write(out_path, excl_defaults=True)

        config_reloaded = ConfigReader.load_file(out_path)

        # dicts are equal if they have the same key value pairs
        assert config.to_dict() == config_reloaded.to_dict()
        assert config == config_reloaded

        # only the actual value should be set not the default
        assert config._defaults['reporting_requirements']._defaults['report_top_ranking_only']
        assert config._defaults['reporting_requirements']._values['report_top_ranking_only']
        assert config._values['reporting_requirements']._defaults['report_top_ranking_only']
        assert not config._values['reporting_requirements']._values['report_top_ranking_only']
        assert not config.reporting_requirements.report_top_ranking_only

    def test_config_to_dict_with_commandline_options_yaml(self, tmpdir):
        config = Config()

        # syntax from __main__.py
        for o in [
            "reporting_requirements.report_top_ranking_only=False"
        ]:
            exec("config." + o)

        out_path = os.path.join(tmpdir, "test_config.yaml")
        config.write_yaml(out_path, excl_defaults=True)

        config_reloaded = ConfigReader.load_file(out_path)

        # dicts are equal if they have the same key value pairs
        assert config.to_dict() == config_reloaded.to_dict()
        assert config == config_reloaded

        # only the actual value should be set not the default
        assert config._defaults['reporting_requirements']._defaults['report_top_ranking_only']
        assert config._defaults['reporting_requirements']._values['report_top_ranking_only']
        assert config._values['reporting_requirements']._defaults['report_top_ranking_only']
        assert not config._values['reporting_requirements']._values['report_top_ranking_only']
        assert not config.reporting_requirements.report_top_ranking_only

    def test_config_to_dict_with_different_enzyme_setups(self, tmpdir):
        # enzyme with digestion rule
        enzyme = Enzyme.trypsinRE

        out_path = os.path.join(tmpdir, "trypsinRE.json")
        enzyme.write(out_path)

        with open(out_path) as f:
            enzyme_reloaded = Enzyme(**json.load(f))

        assert enzyme.to_dict() == enzyme_reloaded.to_dict()
        assert enzyme == enzyme_reloaded
        assert (enzyme.rule == enzyme_reloaded.rule
                == '((?<=[A-Z])[KR]|(?<=^)[KR])(?=[^A-Z]*[A-OQ-Z])')
        assert enzyme.cterminal_of == []
        assert enzyme.nterminal_of == []
        assert enzyme.restraining == []

        # enzyme with cterminal_of digestion
        enzyme = Enzyme.trypsin

        out_path = os.path.join(tmpdir, "trypsin.json")
        enzyme.write(out_path)

        with open(out_path) as f:
            enzyme_reloaded = Enzyme(**json.load(f))

        assert enzyme.to_dict() == enzyme_reloaded.to_dict()
        assert enzyme == enzyme_reloaded
        assert (enzyme.rule == enzyme_reloaded.rule
                == '((?<=[A-Z])(K|R)(?!P)|(?<=^)(K|R)(?!P))')
        assert enzyme.cterminal_of == ['K', 'R']
        assert enzyme.nterminal_of == []
        assert enzyme.restraining == ['P']

        # enzyme with nterminal_of digestion
        enzyme = Enzyme.asp_n

        out_path = os.path.join(tmpdir, "asp_n.json")
        enzyme.write(out_path)

        with open(out_path) as f:
            enzyme_reloaded = Enzyme(**json.load(f))

        assert enzyme.to_dict() == enzyme_reloaded.to_dict()
        assert enzyme == enzyme_reloaded
        assert (enzyme.rule == enzyme_reloaded.rule
                == '(?<=[A-Z])(?=D)')
        assert enzyme.cterminal_of == []
        assert enzyme.nterminal_of == ['D']
        assert enzyme.restraining == []

    def test_config_to_dict_with_losses(self, tmpdir):
        # test if losses with cterm are correctly written and reloaded
        loss = Loss(name='H2O', mass=18.01056027, specificity=['S', 'T', 'D', 'E', 'cterm'])

        out_path = os.path.join(tmpdir, "H2O.json")
        loss.write(out_path)

        with open(out_path) as f:
            loss_reloaded_dict = json.load(f)
            loss_reloaded = Loss(**loss_reloaded_dict)

        assert loss_reloaded_dict['specificity'] == ['S', 'T', 'D', 'E', 'cterm']
        assert loss.to_dict() == loss_reloaded.to_dict()
        assert loss == loss_reloaded

        # test if losses with nterm are correctly written and reloaded
        loss = Loss(name='NH3', mass=17.02654493, specificity=['R', 'K', 'N', 'Q', 'nterm'])

        out_path = os.path.join(tmpdir, "NH3.json")
        loss.write(out_path)

        with open(out_path) as f:
            loss_reloaded_dict = json.load(f)
            loss_reloaded = Loss(**loss_reloaded_dict)

        assert loss_reloaded_dict['specificity'] == ['R', 'K', 'N', 'Q', 'nterm']
        assert loss.to_dict() == loss_reloaded.to_dict()
        assert loss == loss_reloaded

    def test_config_to_dict_with_isotope_rtol(self, tmpdir):
        # test if isotope rtol is derived from ms2_tol
        config = Config(ms2_tol='15ppm')

        out_path = os.path.join(tmpdir, "isotope_rtol.json")
        config.write(out_path)

        with open(out_path) as f:
            config_reloaded = Config(**json.load(f))

        assert_almost_equal(config_reloaded.isotope_config.rtol, 15e-6, decimal=10)
        assert config.to_dict() == config_reloaded.to_dict()
        assert config == config_reloaded

        # test if isotope rtol is set explicitly
        config = Config()
        config.isotope_config.rtol = 5e-6

        out_path = os.path.join(tmpdir, "isotope_rtol.json")
        config.write(out_path)

        with open(out_path) as f:
            config_reloaded = Config(**json.load(f))

        assert_almost_equal(config_reloaded.isotope_config.rtol, 5e-6, decimal=10)
        assert config.to_dict() == config_reloaded.to_dict()
        assert config == config_reloaded

    def test_config_to_dict_with_crosslinkers(self, tmpdir):
        # test if crosslinker  with nterm/cterm is written and reloaded correctly
        crosslinker = Crosslinker(name='EDC',
                                  mass=-18.01056027,
                                  specificity=[["K", "S", "T", "Y", "nterm"], ["E", "D", "cterm"]])

        out_path = os.path.join(tmpdir, "EDC.json")
        crosslinker.write(out_path)

        with open(out_path) as f:
            crosslinker_reloaded_dict = json.load(f)
            crosslinker_reloaded = Crosslinker(**crosslinker_reloaded_dict)

        assert crosslinker.to_dict() == crosslinker_reloaded.to_dict()
        assert crosslinker == crosslinker_reloaded
        assert crosslinker_reloaded.specificity == [["K", "S", "T", "Y"],
                                                    ["E", "D"]]
        assert crosslinker_reloaded_dict['specificity'] == [["K", "S", "T", "Y", "nterm"],
                                                            ["E", "D", "cterm"]]
        assert 'specificity_bonus' not in crosslinker_reloaded_dict

        # test if crosslinker with stubs and bonus is written and reloaded correctly
        crosslinker = Crosslinker(name='DSSO',
                                  mass=158.0038,
                                  specificity=[["K", "S", "T", "Y", "nterm"]],
                                  specificity_bonus=[[0.2, 0, 0, 0, 0.2]],
                                  cleavage_stubs=[
                                      Stub(name='a', mass=54.010565, pairs_with=['s', 't']),
                                      Stub(name='s', mass=103.993200, pairs_with=['a']),
                                      Stub(name='t', mass=85.982635, pairs_with=['a']),
                                  ])

        out_path = os.path.join(tmpdir, "DSSO.json")
        crosslinker.write(out_path)

        with open(out_path) as f:
            crosslinker_reloaded_dict = json.load(f)
            crosslinker_reloaded = Crosslinker(**crosslinker_reloaded_dict)

        assert crosslinker.to_dict() == crosslinker_reloaded.to_dict()
        assert crosslinker == crosslinker_reloaded
