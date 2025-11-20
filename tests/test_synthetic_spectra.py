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

from xicommon.synthetic_spectra import *
from xicommon.config import Config, Crosslinker, FragmentationConfig, Loss
from xicommon import spectra_reader
import os
from xicommon.mock_context import MockContext


def test_create_synthetic_spectrum_mgf(tmpdir):
    # set up config
    config = Config(crosslinker=[Crosslinker.BS3], fragmentation={"add_precursor": False})
    ctx = MockContext(config)

    # set up peptide DB
    peptides = np.array([b'AKT'])
    ctx.setup_peptide_db(peptides)

    expected_mz = [
        36.52583346677, 36.52583346677,
        60.536398042825, 60.536398042825,
        72.04439046677, 72.04439046677,
        120.06551961887999, 120.06551961887999,
        328.702514646535, 328.702514646535,
        352.713079330235, 352.713079330235,
        656.3977528263001, 656.3977528263001,
        704.4188821937, 704.4188821937
    ]

    expected_int = [
        100, 50,
        100, 50,
        100, 50,
        100, 50,
        100, 50,
        100, 50,
        100, 50,
        100, 50
    ]

    pep1_index = 0  # b'AKT'
    pep2_index = 0  # b'AKT'
    link_pos1 = 1
    link_pos2 = 1
    charge = 2

    file_path = os.path.join(tmpdir, 'test.mgf')

    synthetic_spectrum = create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2,
                                                   charge, crosslinker=ctx.config.crosslinker[0],
                                                   context=ctx, isotopes=0)
    create_synthetic_spectra_mgf([synthetic_spectrum], file_path)

    reader = spectra_reader.MGFReader(ctx)
    reader.load(file_path)
    spectrum = next(reader.spectra)

    assert spectrum.precursor['charge'] == charge
    # AKT-AKT z=2 from xi1: 388.23162673677
    np.testing.assert_almost_equal(spectrum.precursor['mz'], 388.23162673677, decimal=4)
    np.testing.assert_almost_equal(spectrum.mz_values, expected_mz, decimal=4)
    np.testing.assert_almost_equal(spectrum.int_values, expected_int)


def test_create_synthetic_spectrum_mgf_mass_delta(tmpdir):
    # set up config
    config = Config(crosslinker=[Crosslinker.BS3])
    ctx = MockContext(config)

    # set up peptide DB
    peptides = np.array([b'AKT'])
    ctx.setup_peptide_db(peptides)

    pep1_index = 0  # b'AKT'
    pep2_index = 0  # b'AKT'
    link_pos1 = 1
    link_pos2 = 1
    charge = 2

    file_path = os.path.join(tmpdir, 'test.mgf')

    synthetic_spectrum = create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2,
                                                   charge, crosslinker=ctx.config.crosslinker[0],
                                                   context=ctx, isotopes=0,
                                                   precursor_delta=-2)
    create_synthetic_spectra_mgf([synthetic_spectrum], file_path)

    reader = spectra_reader.MGFReader(ctx)
    reader.load(file_path)
    spectrum = next(reader.spectra)

    np.testing.assert_almost_equal(spectrum.precursor['mz'], 387.231636222699, decimal=4)


def test_create_synthetic_spectrum_mgf_isotope(tmpdir):
    # set up config
    config = Config(crosslinker=[Crosslinker.BS3], fragmentation={"add_precursor": False})
    ctx = MockContext(config)

    # set up peptide DB
    peptides = np.array([b'AKT'])
    ctx.setup_peptide_db(peptides)

    expected_peaks = np.array([
        # m/z, intensity, sequence , ion, charge, isotope
        (36.52583346677, 100),  # A , b1, z2, i0
        (36.52583346677, 50),  # A , b1, z2, i0
        (37.02751086677, 100),  # A , b1, z2, i1
        (37.02751086677, 50),  # A , b1, z2, i1
        (37.52918826677, 100),  # A , b1, z2, i2
        (37.52918826677, 50),  # A , b1, z2, i2
        (60.53639560177, 100),  # T , y1, z2, i0
        (60.53639560177, 50),  # T , y1, z2, i0
        (61.03807300177, 100),  # T , y1, z2, i1
        (61.03807300177, 50),  # T , y1, z2, i1
        (61.53975040177, 100),  # T , y1, z2, i2
        (61.53975040177, 50),  # T , y1, z2, i2
        (72.04439046677, 100),  # A , b1, z1, i0
        (72.04439046677, 50),  # A , b1, z1, i0
        (73.04774526677, 100),  # A , b1, z1, i1
        (73.04774526677, 50),  # A , b1, z1, i1
        (74.05110006677, 100),  # A , b1, z1, i2
        (74.05110006677, 50),  # A , b1, z1, i2
        (120.06551473677, 100),  # T , y1, z1, i0
        (120.06551473677, 50),  # T , y1, z1, i0
        (121.06886953677, 100),  # T , y1, z1, i1
        (121.06886953677, 50),  # T , y1, z1, i1
        (122.07222433677, 100),  # T , y1, z1, i2
        (122.07222433677, 50),  # T , y1, z1, i2
        (328.70250760177, 100),  # AK + AKT , b2+P, z2, i0
        (328.70250760177, 50),  # AK + AKT , b2+P, z2, i0
        (329.20418500177, 100),  # AK + AKT , b2+P, z2, i1
        (329.20418500177, 50),  # AK + AKT , b2+P, z2, i1
        (329.70586240177, 100),  # AK + AKT , b2+P, z2, i2
        (329.70586240177, 50),  # AK + AKT , b2+P, z2, i2
        (352.71306973677, 100),  # KT + AKT , y2+P, z2, i0
        (352.71306973677, 50),  # KT + AKT , y2+P, z2, i0
        (353.21474713677, 100),  # KT + AKT , y2+P, z2, i1
        (353.21474713677, 50),  # KT + AKT , y2+P, z2, i1
        (353.71642453677, 100),  # KT + AKT , y2+P, z2, i2
        (353.71642453677, 50),  # KT + AKT , y2+P, z2, i2
        (656.39773873677, 100),  # AK + AKT , b2+P, z1, i0
        (656.39773873677, 50),  # AK + AKT , b2+P, z1, i0
        (657.40109353677, 100),  # AK + AKT , b2+P, z1, i1
        (657.40109353677, 50),  # AK + AKT , b2+P, z1, i1
        (658.40444833677, 100),  # AK + AKT , b2+P, z1, i2
        (658.40444833677, 50),  # AK + AKT , b2+P, z1, i2
        (704.41886300677, 100),  # KT + AKT , y2+P, z1, i0
        (704.41886300677, 50),  # KT + AKT , y2+P, z1, i0
        (705.42221780677, 100),  # KT + AKT , y2+P, z1, i1
        (705.42221780677, 50),  # KT + AKT , y2+P, z1, i1
        (706.42557260677, 100),  # KT + AKT , y2+P, z1, i2
        (706.42557260677, 50),  # KT + AKT , y2+P, z1, i2
    ])

    pep1_index = 0  # b'AKT'
    pep2_index = 0  # b'AKT'
    link_pos1 = 1
    link_pos2 = 1
    charge = 2

    file_path = os.path.join(tmpdir, 'test.mgf')

    synthetic_spectrum = create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2,
                                                   charge, crosslinker=ctx.config.crosslinker[0],
                                                   context=ctx, isotopes=2)
    create_synthetic_spectra_mgf([synthetic_spectrum], file_path)

    reader = spectra_reader.MGFReader(ctx)
    reader.load(file_path)
    spectrum = next(reader.spectra)

    assert spectrum.precursor['charge'] == charge
    # sort expected peaks the same way the spectrum is sorted
    sorted_indices = np.argsort(expected_peaks[:, 0])
    expected_peaks = expected_peaks[sorted_indices]
    # AKT-AKT z=2 from xi1: 388.23162673677
    np.testing.assert_almost_equal(spectrum.precursor['mz'], 388.23162673677, decimal=4)
    np.testing.assert_almost_equal(spectrum.mz_values, expected_peaks[:, 0], decimal=3)
    np.testing.assert_almost_equal(spectrum.int_values, expected_peaks[:, 1])


def test_create_synthetic_spectrum_mgf_isotope_precursor(tmpdir):
    # set up config
    config = Config(crosslinker=[Crosslinker.BS3],
                    fragmentation=FragmentationConfig(add_precursor=True))
    ctx = MockContext(config)

    # set up peptide DB
    peptides = np.array([b'AKT'])
    ctx.setup_peptide_db(peptides)

    expected_peaks = np.array([
        # m/z, int, sequence , ion, charge, isotope
        (36.52583346677, 100),  # A , b1, z2, i0
        (36.52583346677, 50),  # A , b1, z2, i0
        (37.02751086677, 100),  # A , b1, z2, i1
        (37.02751086677, 50),  # A , b1, z2, i1
        (60.53639560177, 100),  # T , y1, z2, i0
        (60.53639560177, 50),  # T , y1, z2, i0
        (61.03807300177, 100),  # T , y1, z2, i1
        (61.03807300177, 50),  # T , y1, z2, i1
        (72.04439046677, 100),  # A , b1, z1, i0
        (72.04439046677, 50),  # A , b1, z1, i0
        (73.04774526677, 100),  # A , b1, z1, i1
        (73.04774526677, 50),  # A , b1, z1, i1
        (120.06551473677, 100),  # T , y1, z1, i0
        (120.06551473677, 50),  # T , y1, z1, i0
        (121.06886953677, 100),  # T , y1, z1, i1
        (121.06886953677, 50),  # T , y1, z1, i1
        (328.70250760177, 100),  # AK + AKT , b2+P, z2, i0
        (328.70250760177, 50),  # AK + AKT , b2+P, z2, i0
        (329.20418500177, 100),  # AK + AKT , b2+P, z2, i1
        (329.20418500177, 50),  # AK + AKT , b2+P, z2, i1
        (352.71306973677, 100),  # KT + AKT , y2+P, z2, i0
        (352.71306973677, 50),  # KT + AKT , y2+P, z2, i0
        (353.21474713677, 100),  # KT + AKT , y2+P, z2, i1
        (353.21474713677, 50),  # KT + AKT , y2+P, z2, i1
        (388.231626736879, 100),  # AKT + AKT, P, z2, i0
        (388.733304136879, 100),  # AKT + AKT, P, z2, i1
        (656.39773873677, 100),  # AK + AKT , b2+P, z1, i0
        (656.39773873677, 50),  # AK + AKT , b2+P, z1, i0
        (657.40109353677, 100),  # AK + AKT , b2+P, z1, i1
        (657.40109353677, 50),  # AK + AKT , b2+P, z1, i1
        (704.41886300677, 100),  # KT + AKT , y2+P, z1, i0
        (704.41886300677, 50),  # KT + AKT , y2+P, z1, i0
        (705.42221780677, 100),  # KT + AKT , y2+P, z1, i1
        (705.42221780677, 50),  # KT + AKT , y2+P, z1, i1
        (775.455977006879, 100),  # AKT + AKT, P, z1, i0
        (776.459331806879, 100),  # AKT + AKT, P, z1, i1
    ])

    pep1_index = 0  # b'AKT'
    pep2_index = 0  # b'AKT'
    link_pos1 = 1
    link_pos2 = 1
    charge = 2

    file_path = os.path.join(tmpdir, 'test.mgf')

    synthetic_spectrum = create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2,
                                                   charge, crosslinker=ctx.config.crosslinker[0],
                                                   context=ctx, isotopes=1)
    create_synthetic_spectra_mgf([synthetic_spectrum], file_path)

    reader = spectra_reader.MGFReader(ctx)
    reader.load(file_path)
    spectrum = next(reader.spectra)

    assert spectrum.precursor['charge'] == charge
    # sort expected peaks the same way the spectrum is sorted
    sorted_indices = np.argsort(expected_peaks[:, 0])
    expected_peaks = expected_peaks[sorted_indices]
    # AKT-AKT z=2 from xi1: 388.23162673677
    np.testing.assert_almost_equal(spectrum.precursor['mz'], 388.23162673677, decimal=4)
    np.testing.assert_almost_equal(spectrum.mz_values, expected_peaks[:, 0], decimal=3)
    np.testing.assert_almost_equal(spectrum.int_values, expected_peaks[:, 1])


def test_create_synthetic_spectrum_mgf_linear(tmpdir):
    # set up config
    config = Config(fragmentation=FragmentationConfig(add_precursor=True))
    ctx = MockContext(config)

    # set up peptide DB
    peptides = np.array([b'STAY'])
    ctx.setup_peptide_db(peptides)

    expected_mz = [
        30.01795246677,  # b1  z3   S
        44.52329046677,  # b1  z2   S
        61.36523955677,  # y1  z3   Y
        63.7005118001033,  # b2  z3   ST
        85.04427755677,  # y2  z3   AY
        87.3795498001033,  # b3  z3   STA
        88.03930446677,  # b1  z1   S
        91.54422110177,  # y1  z2   Y
        95.04712946677,  # b2  z2   ST
        118.726836890103,  # y3  z3   TAY
        127.06277810177,  # y2  z2   AY
        130.56568646677,  # b3  z2   STA
        147.737512890103,  # P  z3   STAY
        177.58661710177,  # y3  z2   TAY
        182.08116573677,  # y1  z1   Y
        189.08698246677,  # b2  z1   ST
        221.10263110177,  # P  z2   STAY
        253.11827973677,  # y2  z1   AY
        260.12409646677,  # b3  z1   STA
        354.16595773677,  # y3  z1   TAY
        441.19798573677  # P  z1   STAY
    ]

    expected_int = [100] * len(expected_mz)

    pep1_index = 0  # b'STAY'
    pep2_index = -1  # no 2nd peptide
    link_pos1 = -1  # no crosslink
    link_pos2 = -1  # no crosslink
    charge = 3
    crosslinker = None

    file_path = os.path.join(tmpdir, 'test.mgf')

    synthetic_spectrum = create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2,
                                                   charge, crosslinker, ctx, isotopes=0)
    create_synthetic_spectra_mgf([synthetic_spectrum], file_path)

    reader = spectra_reader.MGFReader(ctx)
    reader.load(file_path)
    spectrum = next(reader.spectra)

    assert spectrum.precursor['charge'] == charge
    # STAY z3 from xi1: 147.737512890103
    np.testing.assert_almost_equal(spectrum.precursor['mz'], 147.737512890103, decimal=4)
    np.testing.assert_almost_equal(spectrum.mz_values, expected_mz, decimal=4)
    np.testing.assert_almost_equal(spectrum.int_values, expected_int)


def test_create_synthetic_spectrum_mgf_noncovalent(tmpdir):
    # set up config
    config = Config(fragmentation=FragmentationConfig(add_precursor=True))
    ctx = MockContext(config)

    # set up peptide DB
    peptides = np.array([b'HQME', b'STAY'])
    ctx.setup_peptide_db(peptides)

    expected_peaks = np.array([
        (30.01795246677, 100),  # b1     z3     S
        (44.52329046677, 100),  # b1     z2     S
        (46.69358046677, 50),  # b1     z3     H
        (50.0249942234367, 50),  # y1     z3     E
        (61.36523955677, 100),  # y1     z3     Y
        (63.7005118001033, 100),  # b2     z3     ST
        (69.53673246677, 50),  # b1     z2     H
        (74.53385310177, 50),  # y1     z2     E
        (85.04427755677, 100),  # y2     z3     AY
        (87.3795498001033, 100),  # b3     z3     STA
        (88.03930446677, 100),  # b1     z1     S
        (89.3797731334367, 50),  # b2     z3     HQ
        (91.54422110177, 100),  # y1     z2     Y
        (93.7051558901034, 50),  # y2     z3     ME
        (95.04712946677, 100),  # b2     z2     ST
        (118.726836890103, 100),  # y3     z3     TAY
        (127.06277810177, 100),  # y2     z2     AY
        (130.56568646677, 100),  # b3     z2     STA
        (133.059934800103, 50),  # b3     z3     HQM
        (133.56602146677, 50),  # b2     z2     HQ
        (136.39134855677, 50),  # y3     z3     QME
        (138.06618846677, 50),  # b1     z1     H
        (140.05409560177, 50),  # y2     z2     ME
        (147.737512890103, 100),  # P     z3     STAY
        (148.06042973677, 50),  # y1     z1     E
        (177.58661710177, 100),  # y3     z2     TAY
        (182.07765255677, 50),  # P     z3     HQME
        (182.08116573677, 100),  # y1     z1     Y
        (189.08698246677, 100),  # b2     z1     ST
        (199.08626396677, 50),  # b3     z2     HQM
        (204.08338460177, 50),  # y3     z2     QME
        (221.10263110177, 100),  # P     z2     STAY
        (253.11827973677, 100),  # y2     z1     AY
        (260.12409646677, 100),  # b3     z1     STA
        (266.12476646677, 50),  # b2     z1     HQ
        (272.61284060177, 50),  # P     z2     HQME
        (279.10091473677, 50),  # y2     z1     ME
        (354.16595773677, 100),  # y3     z1     TAY
        (397.16525146677, 50),  # b3     z1     HQM
        (407.15949273677, 50),  # y3     z1     QME
        (441.19798573677, 100),  # P     z1     STAY
        (544.21840473677, 50),  # P     z1     HQME
    ])

    pep1_index = 1  # STAY
    pep2_index = 0  # HQME
    link_pos1 = -1  # no crosslink
    link_pos2 = -1  # no crosslink
    charge = 3
    crosslinker = None

    file_path = os.path.join(tmpdir, 'test.mgf')
    synthetic_spectrum = create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2,
                                                   charge, crosslinker, ctx, isotopes=0)
    create_synthetic_spectra_mgf([synthetic_spectrum], file_path)

    reader = spectra_reader.MGFReader(ctx)
    reader.load(file_path)
    spectrum = next(reader.spectra)

    assert spectrum.precursor['charge'] == charge
    # STAY HQME z3 from xi1: 328.80788898010337
    np.testing.assert_almost_equal(spectrum.precursor['mz'], 328.80788898010337, decimal=4)
    np.testing.assert_almost_equal(spectrum.mz_values, expected_peaks[:, 0], decimal=4)
    np.testing.assert_almost_equal(spectrum.int_values, expected_peaks[:, 1])


def test_create_synthetic_spectrum_mgf_losses(tmpdir):
    """Test creating a synthetic spectrum with losses and isotopes."""
    # set up config
    water_loss = Loss(name='H2O', composition="H2O1", specificity=['S', 'T', 'E', 'D', 'cterm'])
    ammonia_loss = Loss(name='NH3', composition="N1H3", specificity=['R', 'K', 'N', 'Q', 'nterm'])

    config = Config(
        crosslinker=[Crosslinker.BS3],
        fragmentation=FragmentationConfig(add_precursor=True, losses=[water_loss, ammonia_loss],
                                          max_nloss=2)
    )
    ctx = MockContext(config)

    # set up peptide DB
    peptides = np.array([b'MAGE', b'RNG'])
    ctx.setup_peptide_db(peptides)

    c12c13 = const.C12C13_MASS_DIFF

    expected_peaks = np.array([
        # (mz, int),  # frag, charge, sequence
        (29.51800846677, 100),  # y1_H20x1 z2 G
        (29.51800846677 + c12c13 / 2, 100),  # y1_H20x1 z2 G iso1
        (38.52328860177, 100),  # y1 z2 G
        (38.52328860177 + c12c13 / 2, 100),  # y1 z2 G iso1
        (56.52329283177, 50),  # y1_H20x2 z2 E
        (56.52329283177 + c12c13 / 2, 50),  # y1_H20x2 z2 E iso1
        (58.02874046677, 100),  # y1_H20x1 z1 G
        (58.02874046677 + c12c13 / 1, 100),  # y1_H20x1 z1 G iso1
        (65.52857296677, 50),  # y1_H20x1 z2 E
        (65.52857296677 + c12c13 / 2, 50),  # y1_H20x1 z2 E iso1
        (74.53385310177, 50),  # y1 z2 E
        (74.53385310177 + c12c13 / 2, 50),  # y1 z2 E iso1
        (76.03930073677, 100),  # y1 z1 G
        (76.03930073677 + c12c13 / 1, 100),  # y1 z1 G iso1
        (78.02619950177, 100),  # y2_H20x1_NH3x1 z2 NG
        (78.02619950177 + c12c13 / 2, 100),  # y2_H20x1_NH3x1 z2 NG iso1
        (85.03402483177, 50),  # y2_H20x2 z2 GE
        (85.03402483177 + c12c13 / 2, 50),  # y2_H20x2 z2 GE iso1
        (86.53947196677, 100),  # y2_H20x1 z2 NG
        (86.53947196677 + c12c13 / 2, 100),  # y2_H20x1 z2 NG iso1
        (87.03147963677, 100),  # y2_NH3x1 z2 NG
        (87.03147963677 + c12c13 / 2, 100),  # y2_NH3x1 z2 NG iso1
        (94.03930496677, 50),  # y2_H20x1 z2 GE
        (94.03930496677 + c12c13 / 2, 50),  # y2_H20x1 z2 GE iso1
        (95.54475210177, 100),  # y2 z2 NG
        (95.54475210177 + c12c13 / 2, 100),  # y2 z2 NG iso1
        (103.04458510177, 50),  # y2 z2 GE
        (103.04458510177 + c12c13 / 2, 50),  # y2 z2 GE iso1
        (112.03930919677, 50),  # y1_H20x2 z1 E
        (112.03930919677 + c12c13 / 1, 50),  # y1_H20x2 z1 E iso1
        (120.55258183177, 50),  # y3_H20x2 z2 AGE
        (120.55258183177 + c12c13 / 2, 50),  # y3_H20x2 z2 AGE iso1
        (129.55786196677, 50),  # y3_H20x1 z2 AGE
        (129.55786196677 + c12c13 / 2, 50),  # y3_H20x1 z2 AGE iso1
        (130.04986946677, 50),  # y1_H20x1 z1 E
        (130.04986946677 + c12c13 / 1, 50),  # y1_H20x1 z1 E iso1
        (138.56314210177, 50),  # y3 z2 AGE
        (138.56314210177 + c12c13 / 2, 50),  # y3 z2 AGE iso1
        (148.06042973677, 50),  # y1 z1 E
        (148.06042973677 + c12c13 / 1, 50),  # y1 z1 E iso1
        (155.04512253677, 100),  # y2_H20x1_NH3x1 z1 NG
        (155.04512253677 + c12c13 / 1, 100),  # y2_H20x1_NH3x1 z1 NG iso1
        (169.06077319677, 50),  # y2_H20x2 z1 GE
        (169.06077319677 + c12c13 / 1, 50),  # y2_H20x2 z1 GE iso1
        (172.07166746677, 100),  # y2_H20x1 z1 NG
        (172.07166746677 + c12c13 / 1, 100),  # y2_H20x1 z1 NG iso1
        (173.05568280677, 100),  # y2_NH3x1 z1 NG
        (173.05568280677 + c12c13 / 1, 100),  # y2_NH3x1 z1 NG iso1
        (187.07133346677, 50),  # y2_H20x1 z1 GE
        (187.07133346677 + c12c13 / 1, 50),  # y2_H20x1 z1 GE iso1
        (190.08222773677, 100),  # y2 z1 NG
        (190.08222773677 + c12c13 / 1, 100),  # y2 z1 NG iso1
        (205.08189373677, 50),  # y2 z1 GE
        (205.08189373677 + c12c13 / 1, 50),  # y2 z1 GE iso1
        (240.09788719677, 50),  # y3_H20x2 z1 AGE
        (240.09788719677 + c12c13 / 1, 50),  # y3_H20x2 z1 AGE iso1
        (258.10844746677, 50),  # y3_H20x1 z1 AGE
        (258.10844746677 + c12c13 / 1, 50),  # y3_H20x1 z1 AGE iso1
        (276.11900773677, 50),  # y3 z1 AGE
        (276.11900773677 + c12c13 / 1, 50),  # y3 z1 AGE iso1
        (290.63103250177, 50),  # b1+P_H20x1_NH3x1 z2 M + RNG
        (290.63103250177 + c12c13 / 2, 50),  # b1+P_H20x1_NH3x1 iso1
        (291.12304017177, 50),  # b1+P_NH3x2 z2 M + RNG
        (291.12304017177 + c12c13 / 2, 50),  # b1+P_NH3x2 z2 M + RNG iso1
        (299.14430496677, 50),  # b1+P_H20x1  z2 M + RNG
        (299.14430496677 + c12c13 / 2, 50),  # b1+P_H20x1  z2 M + RNG iso1
        (299.63631263677, 50),  # b1+P_NH3x1 z2 M + RNG
        (299.63631263677 + c12c13 / 2, 50),  # b1+P_NH3x1 z2 M + RNG iso1
        (308.14958510177, 50),  # b1+P z2 M + RNG
        (308.14958510177 + c12c13 / 2, 50),  # b1+P z2 M + RNG iso1
        (326.14958950177, 50),  # b2+P_H20x1_NH3x1 z2 MA + RNG
        (326.14958950177 + c12c13 / 2, 50),  # b2+P_H20x1_NH3x1 z2 MA + RNG iso1
        (326.64159717177, 50),  # b2+P_NH3x2 z2 MA + RNG
        (326.64159717177 + c12c13 / 2, 50),  # b2+P_NH3x2 z2 MA + RNG iso1
        (333.64942250177, 100),  # b1+P_H20x1_NH3x1 z2 R + MAGE
        (333.64942250177 + c12c13 / 2, 100),  # b1+P_H20x1_NH3x1 z2 R + MAGE iso1
        (334.14143017177, 100),  # b1+P_NH3x2 z2 R + MAGE
        (334.14143017177 + c12c13 / 2, 100),  # b1+P_NH3x2 z2 R + MAGE iso1
        (334.66286196677, 50),  # b2+P_H20x1  z2 MA + RNG
        (334.66286196677 + c12c13 / 2, 50),  # b2+P_H20x1  z2 MA + RNG iso1
        (335.15486963677, 50),  # b2+P_NH3x1 z2 MA + RNG
        (335.15486963677 + c12c13 / 2, 50),  # b2+P_NH3x1 z2 MA + RNG iso1
        (342.16269496677, 100),  # b1+P_H20x1 z2 R + MAGE
        (342.16269496677 + c12c13 / 2, 100),  # b1+P_H20x1 z2 R + MAGE iso1
        (342.65470263677, 100),  # b1+P_NH3x1 z2 R + MAGE
        (342.65470263677 + c12c13 / 2, 100),  # b1+P_NH3x1 z2 R + MAGE iso1
        (343.66814210177, 50),  # b2+P z2 MA + RNG
        (343.66814210177 + c12c13 / 2, 50),  # b2+P z2 MA + RNG iso1
        (351.16797510177, 100),  # b1+P z2 R + MAGE
        (351.16797510177 + c12c13 / 2, 100),  # b1+P z2 R + MAGE iso1
        (354.66032150177, 50),  # b3+P_H20_NH3x1 z2 MAG + RNG
        (354.66032150177 + c12c13 / 2, 50),  # b3+P_H20_NH3x1 z2 MAG + RNG iso1
        (355.15232917177, 50),  # b3+P_NH3x2 z2 MAG + RNG
        (355.15232917177 + c12c13 / 2, 50),  # b3+P_NH3x2 z2 MAG + RNG iso1
        (363.17359396677, 50),  # b3+P_H20x1 z2 MAG + RNG
        (363.17359396677 + c12c13 / 2, 50),  # b3+P_H20x1 z2 MAG + RNG iso1
        (363.66560163677, 50),  # b3+P_NH3x1 z2 MAG + RNG
        (363.66560163677 + c12c13 / 2, 50),  # b3+P_NH3x1 z2 MAG + RNG iso1
        (372.17887410177, 50),  # b3+P z2 MAG + RNG
        (372.17887410177 + c12c13 / 2, 50),  # b3+P z2 MAG + RNG iso1
        (390.67088600177, 100),  # b2+P_H20x1_NH3x1 z2 RN + MAGE
        (390.67088600177 + c12c13 / 2, 100),  # b2+P_H20x1_NH3x1 z2 RN + MAGE iso1
        (391.16289367177, 100),  # b2+P_NH3x2 z2 RN + MAGE
        (391.16289367177 + c12c13 / 2, 100),  # b2+P_NH3x2 z2 RN + MAGE iso1
        (399.18415846677, 100),  # b2+P_H20x1 z2 RN + MAGE
        (399.18415846677 + c12c13 / 2, 100),  # b2+P_H20x1 z2 RN + MAGE iso1
        (399.67616613677, 100),  # b2+P_NH3x1 z2 RN + MAGE
        (399.67616613677 + c12c13 / 2, 100),  # b2+P_NH3x1 z2 RN + MAGE iso1
        (408.18943860177, 100),  # b2+P z2 RN + MAGE
        (408.18943860177 + c12c13 / 2, 100),  # b2+P z2 RN + MAGE iso1
        (427.69489046677, 100),  # P+P_H20x2 z2 RNG + MAGE
        (427.69489046677 + c12c13 / 2, 100),  # P+P_H20x2 z2 RNG + MAGE iso1
        (428.18689813677, 100),  # P+P_H20x1_NH3x1 z2 RNG + MAGE
        (428.18689813677 + c12c13 / 2, 100),  # P+P_H20x1_NH3x1 z2 RNG + MAGE iso1
        (428.67890580677, 100),  # P+P_NH3x2 z2 RNG + MAGE
        (428.67890580677 + c12c13 / 2, 100),  # P+P_NH3x2 z2 RNG + MAGE iso1
        (436.70017060177, 100),  # P+P_H20x1 z2 RNG + MAGE
        (436.70017060177 + c12c13 / 2, 100),  # P+P_H20x1 z2 RNG + MAGE iso1
        (437.19217827177, 100),  # P+P_NH3x1 z2 RNG + MAGE
        (437.19217827177 + c12c13 / 2, 100),  # P+P_NH3x1 z2 RNG + MAGE iso1
        (445.70545073677, 100),  # P+P z2 RNG + MAGE
        (445.70545073677 + c12c13 / 2, 100),  # P+P z2 RNG + MAGE iso1
        (580.25478853677, 50),  # b1+P_H20_NH3x1 z1 M + RNG
        (580.25478853677 + c12c13 / 1, 50),  # b1+P_H20_NH3x1 z1 M + RNG iso1
        (581.23880387677, 50),  # b1+P_NH3x2 z1 M + RNG
        (581.23880387677 + c12c13 / 1, 50),  # b1+P_NH3x2 z1 M + RNG iso1
        (597.28133346677, 50),  # b1+P_H20x1 z1 M + RNG
        (597.28133346677 + c12c13 / 1, 50),  # b1+P_H20x1 z1 M + RNG iso1
        (598.26534880677, 50),  # b1+P_NH3x1 z1 M + RNG
        (598.26534880677 + c12c13 / 1, 50),  # b1+P_NH3x1 z1 M + RNG iso1
        (615.29189373677, 50),  # b1+P z1 M + RNG
        (615.29189373677 + c12c13 / 1, 50),  # b1+P z1 M + RNG iso1
        (651.29190253677, 50),  # b2+P_H20x1_NH3x1 z1 MA + RNG
        (651.29190253677 + c12c13 / 1, 50),  # b2+P_H20x1_NH3x1 z1 MA + RNG iso1
        (652.27591787677, 50),  # b2+P_NH3x2 z1 MA + RNG
        (652.27591787677 + c12c13 / 1, 50),  # b2+P_NH3x2 z1 MA + RNG iso1
        (666.29156853677, 100),  # b1+P_H20x1_NH3x1 z1 R + MAGE
        (666.29156853677 + c12c13 / 1, 100),  # b1+P_H20x1_NH3x1 z1 R + MAGE iso1
        (667.27558387677, 100),  # b1+P_NH3x2 z1 R + MAGE
        (667.27558387677 + c12c13 / 1, 100),  # b1+P_NH3x2 z1 R + MAGE iso1
        (668.31844746677, 50),  # b2+P_H20x1 z1 MA + RNG
        (668.31844746677 + c12c13 / 1, 50),  # b2+P_H20x1 z1 MA + RNG iso1
        (669.30246280677, 50),  # b2+P_NH3x1 z1 MA + RNG
        (669.30246280677 + c12c13 / 1, 50),  # b2+P_NH3x1 z1 MA + RNG iso1
        (683.31811346677, 100),  # b1+P_H20x1 z1 R + MAGE
        (683.31811346677 + c12c13 / 1, 100),  # b1+P_H20x1 z1 R + MAGE iso1
        (684.30212880677, 100),  # b1+P_NH3x1 z1 R + MAGE
        (684.30212880677 + c12c13 / 1, 100),  # b1+P_NH3x1 z1 R + MAGE iso1
        (686.32900773677, 50),  # b2+P z1 MA + RNG
        (686.32900773677 + c12c13 / 1, 50),  # b2+P z1 MA + RNG iso1
        (701.32867373677, 100),  # b1+P z1 R + MAGE
        (701.32867373677 + c12c13 / 1, 100),  # b1+P z1 R + MAGE iso1
        (708.31336653677, 50),  # b3+P_H20x1_NH3x1 z1 MAG + RNG
        (708.31336653677 + c12c13 / 1, 50),  # b3+P_H20x1_NH3x1 z1 MAG + RNG iso1
        (709.29738187677, 50),  # b3+P_NH3x2 z1 MAG + RNG
        (709.29738187677 + c12c13 / 1, 50),  # b3+P_NH3x2 z1 MAG + RNG iso1
        (725.33991146677, 50),  # b3+P_H20x1 z1 MAG + RNG
        (725.33991146677 + c12c13 / 1, 50),  # b3+P_H20x1 z1 MAG + RNG iso1
        (726.32392680677, 50),  # b3+P_NH3x1 z1 MAG + RNG
        (726.32392680677 + c12c13 / 1, 50),  # b3+P_NH3x1 z1 MAG + RNG iso1
        (743.35047173677, 50),  # b3+P z1 MAG + RNG
        (743.35047173677 + c12c13 / 1, 50),  # b3+P z1 MAG + RNG iso1
        (780.33449553677, 100),  # b2+P_H20x1_NH3x1 z1 RN + MAGE
        (780.33449553677 + c12c13 / 1, 100),  # b2+P_H20x1_NH3x1 z1 RN + MAGE iso1
        (781.31851087677, 100),  # b2+P_NH3x2 z1 RN + MAGE
        (781.31851087677 + c12c13 / 1, 100),  # b2+P_NH3x2 z1 RN + MAGE iso1
        (797.36104046677, 100),  # b2+P_H20x1 z1 RN + MAGE
        (797.36104046677 + c12c13 / 1, 100),  # b2+P_H20x1 z1 RN + MAGE iso1
        (798.34505580677, 100),  # b2+P_NH3x1 z1 RN + MAGE
        (798.34505580677 + c12c13 / 1, 100),  # b2+P_NH3x1 z1 RN + MAGE iso1
        (815.37160073677, 100),  # b2+P z1 RN + MAGE
        (815.37160073677 + c12c13 / 1, 100),  # b2+P z1 RN + MAGE iso1
        (854.38250446677, 100),  # P+P_H20x2 z1 RNG + MAGE
        (854.38250446677 + c12c13 / 1, 100),  # P+P_H20x2 z1 RNG + MAGE iso1
        (855.36651980677, 100),  # P+P_H20x1_NH3x1 z1 RNG + MAGE
        (855.36651980677 + c12c13 / 1, 100),  # P+P_H20x1_NH3x1 z1 RNG + MAGE iso1
        (856.35053514677, 100),  # P+P_NH3x2 z1 RNG + MAGE
        (856.35053514677 + c12c13 / 1, 100),  # P+P_NH3x2 z1 RNG + MAGE iso1
        (872.39306473677, 100),  # P+P_H20x1 z1 RNG + MAGE
        (872.39306473677 + c12c13 / 1, 100),  # P+P_H20x1 z1 RNG + MAGE iso1
        (873.37708007677, 100),  # P+P_NH3x1 z1 RNG + MAGE
        (873.37708007677 + c12c13 / 1, 100),  # P+P_NH3x1 z1 RNG + MAGE iso1
        (890.40362500677, 100),  # P+P z1 RNG + MAGE
        (890.40362500677 + c12c13 / 1, 100),  # P+P z1 RNG + MAGE iso1
    ], dtype=[('mz', np.float64), ('int', np.float64)])
    expected_peaks.sort(order='mz')

    pep1_index = 1  # RNG
    pep2_index = 0  # MAGE
    link_pos1 = 0
    link_pos2 = 0
    charge = 2

    file_path = os.path.join(tmpdir, 'test.mgf')
    synthetic_spectrum = create_synthetic_spectrum(pep1_index, pep2_index, link_pos1, link_pos2,
                                                   charge, ctx.config.crosslinker[0], ctx,
                                                   isotopes=1)
    create_synthetic_spectra_mgf([synthetic_spectrum], file_path)

    reader = spectra_reader.MGFReader(ctx)
    reader.load(file_path)
    spectrum = next(reader.spectra)

    assert spectrum.precursor['charge'] == charge
    # RNG-MAGE z2 from xi1: 445.70545073677
    np.testing.assert_almost_equal(spectrum.precursor['mz'], 445.70545073677, decimal=4)
    np.testing.assert_almost_equal(spectrum.mz_values, expected_peaks['mz'], decimal=4)
    np.testing.assert_almost_equal(spectrum.int_values, expected_peaks['int'])
