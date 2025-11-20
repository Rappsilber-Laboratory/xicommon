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

from xicommon.spectra_reader import *
from xicommon import synthetic_spectra
from xicommon.filters import IsotopeDetector, IsotopeReducer, ChargeReducer
from xicommon.config import Config, Crosslinker
from xicommon.fragment_peptides import fragment_crosslinked_peptide_pair
import xicommon.dtypes as dtypes
from xicommon.fragmentation import spread_charges
import pytest
from numpy.testing import assert_array_equal, assert_equal, assert_allclose
from xicommon.mock_context import MockContext
import os


test_mgf_expected = {
    'precursors': [{'mz': 983.6124, 'charge': 2, 'intensity': 1622453.2378},
                   {'mz': 235.6236, 'charge': 3, 'intensity': 3562467.2343}],
    'mz_arrays': [(846.60, 846.80, 847.60), (345.10, 370.20, 460.20)],
    'int_arrays': [(73, 44, 67), (237, 128, 108)],
    'scan_ids': ['TestFileName controllerType=0 controllerNumber=1 scan=1',
                 'TestFileName controllerType=0 controllerNumber=1 scan=2'],
    'retention_times': [60, 123],
    'file_names': ['test.mgf', 'test.mgf'],
    'run_names': ['TestFileName', 'TestFileName'],
    'scan_numbers': [1, 2],
    'scan_indices': [0, 1],
}

test2_mgf_expected = {
    'precursors': [{'mz': 1000.1, 'charge': 3, 'intensity': 42.314},
                   {'mz': 9000.1, 'charge': 2, 'intensity': 131415.666}],
    'mz_arrays': [(345.12, 456.23), (567.34, 678.45)],
    'int_arrays': [(34, 45), (56, 67)],
    'scan_ids': ['TestFileName2 controllerType=0 controllerNumber=1 scan=10',
                 'TestFileName2 controllerType=0 controllerNumber=1 scan=20'],
    'retention_times': [601, 1234],
    'file_names': ['test2.mgf', 'test2.mgf'],
    'run_names': ['TestFileName2', 'TestFileName2'],
    'scan_numbers': [10, 20],
    'scan_indices': [0, 1],
}

test3_mgf_expected = {
    'precursors': [{'mz': 2000.2, 'charge': 5, 'intensity': 542.6},
                   {'mz': 2646.15, 'charge': 6, 'intensity': 37438.8346}],
    'mz_arrays': [(725.2784, 746.2975, 1545.6657), (134.324, 678.632)],
    'int_arrays': [(3932.34, 487534.567, 246547.32), (246.68, 245.54)],
    'scan_ids': ['TestFileName3 controllerType=0 controllerNumber=1 scan=71',
                 'TestFileName3 controllerType=0 controllerNumber=1 scan=262'],
    'retention_times': [5498, 12326],
    'file_names': ['test3.mgf', 'test3.mgf'],
    'run_names': ['TestFileName3', 'TestFileName3'],
    'scan_numbers': [71, 262],
    'scan_indices': [0, 1],
}

fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures', 'spectra')


def check_spectra_reader(reader, expected):
    for i, spectrum in enumerate(reader.spectra):
        assert spectrum.precursor['mz'] == expected['precursors'][i]['mz']
        assert spectrum.precursor['charge'] == expected['precursors'][i]['charge']
        assert spectrum.precursor['intensity'] == expected['precursors'][i]['intensity']
        assert_equal(spectrum.mz_values, expected['mz_arrays'][i])
        assert_equal(spectrum.int_values, expected['int_arrays'][i])
        assert spectrum.scan_id == expected['scan_ids'][i]
        assert spectrum.rt == expected['retention_times'][i]
        assert spectrum.file_name == expected['file_names'][i]
        assert spectrum.run_name == expected['run_names'][i]
        assert spectrum.scan_number == expected['scan_numbers'][i]
        assert spectrum.scan_index == expected['scan_indices'][i]


def check_spectra_reader_raw(scan_number, reader, expected):

    reader_values = reader._convert_spectrum(scan_number)
    assert reader_values.precursor['mz'] == expected['precursor']['mz']
    assert reader_values.precursor['charge'] == expected['precursor']['charge']
    assert reader_values.precursor['intensity'] == expected['precursor']['intensity']
    assert_equal(reader_values.mz_values, expected['mz_array'])
    assert_equal(reader_values.int_values, expected['int_array'])
    assert reader_values.scan_id == expected['scan_id']
    assert reader_values.rt == expected['retention_time']
    assert reader_values.file_name == expected['file_name']
    assert reader_values.run_name == expected['run_name']
    assert reader_values.scan_number == expected['scan_number']
    assert reader_values.scan_index == expected['scan_index']


def test_peaklist_wrapper(tmpdir):
    ctx = MockContext()
    wrapper = PeakListWrapper(ctx)

    # tar file containing test.mgf and test2.mgf
    tar_file = os.path.join(fixtures_dir, 'test.tar.gz')

    pl_file = os.path.join(fixtures_dir, 'test3.mgf')

    # test tar archive with additional mgf file
    wrapper.load([tar_file, pl_file])
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 6
    assert len(wrapper.readers) == 3
    # check test.mgf
    assert wrapper.readers[0].file_name == 'test.mgf'
    assert wrapper.readers[0].source_path == os.path.join(tar_file, 'test.mgf')
    check_spectra_reader(wrapper.readers[0], test_mgf_expected)
    # check test2.mgf
    assert wrapper.readers[1].file_name == 'test2.mgf'
    assert wrapper.readers[1].source_path == os.path.join(tar_file, 'test2.mgf')
    check_spectra_reader(wrapper.readers[1], test2_mgf_expected)
    # check test3.mgf
    assert wrapper.readers[2].file_name == 'test3.mgf'
    assert wrapper.readers[2].source_path == pl_file
    check_spectra_reader(wrapper.readers[2], test3_mgf_expected)

    # test zip archive with additional mgf file
    # zip file containing test.mgf and test2.mgf
    zip_file = os.path.join(fixtures_dir, 'test.zip')
    wrapper.load([zip_file, pl_file])
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 6
    assert len(wrapper.readers) == 3
    # test the same file but as deflate64 compressed
    zip_file = os.path.join(fixtures_dir, 'test64.zip')
    wrapper.load([zip_file, pl_file])
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 6
    assert len(wrapper.readers) == 3
    # check test.mgf
    assert wrapper.readers[0].file_name == 'test.mgf'
    assert wrapper.readers[0].source_path == os.path.join(zip_file, 'test.mgf')
    check_spectra_reader(wrapper.readers[0], test_mgf_expected)
    # check test2.mgf
    assert wrapper.readers[1].file_name == 'test2.mgf'
    assert wrapper.readers[1].source_path == os.path.join(zip_file, 'test2.mgf')
    check_spectra_reader(wrapper.readers[1], test2_mgf_expected)
    # check test3.mgf
    assert wrapper.readers[2].file_name == 'test3.mgf'
    assert wrapper.readers[2].source_path == pl_file
    check_spectra_reader(wrapper.readers[2], test3_mgf_expected)

    # test archives with test2.mgf in subdir
    subdir_tar_file = os.path.join(fixtures_dir, 'subdir_test.tar.gz')
    wrapper.load(subdir_tar_file)
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 4
    assert len(wrapper.readers) == 2
    assert wrapper.readers[0].file_name == 'test2.mgf'
    assert wrapper.readers[0].source_path == os.path.join(subdir_tar_file, 'subdir', 'test2.mgf')
    check_spectra_reader(wrapper.readers[0], test2_mgf_expected)
    assert wrapper.readers[1].file_name == 'test.mgf'
    assert wrapper.readers[1].source_path == os.path.join(subdir_tar_file, 'test.mgf')
    check_spectra_reader(wrapper.readers[1], test_mgf_expected)

    subdir_zip_file = os.path.join(fixtures_dir, 'subdir_test.zip')
    wrapper.load(subdir_zip_file)
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 4
    assert len(wrapper.readers) == 2
    assert wrapper.readers[0].file_name == 'test2.mgf'
    assert wrapper.readers[0].source_path == os.path.join(subdir_zip_file, 'subdir', 'test2.mgf')
    check_spectra_reader(wrapper.readers[0], test2_mgf_expected)
    assert wrapper.readers[1].file_name == 'test.mgf'
    assert wrapper.readers[1].source_path == os.path.join(subdir_zip_file, 'test.mgf')
    check_spectra_reader(wrapper.readers[1], test_mgf_expected)

    empty_dir = os.path.join(tmpdir, 'empty_dir')
    os.makedirs(empty_dir)
    with pytest.raises(ValueError):
        wrapper.load(empty_dir)

    unknown_file = os.path.join(tmpdir, 'nospectra.dummy')
    open(unknown_file, 'a').close()
    with pytest.raises(ValueError):
        wrapper.load(unknown_file)

    wrapper.load(fixtures_dir)
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 161

    nofile = os.path.join(fixtures_dir, 'non-existing-file.mgf')
    with pytest.raises(ValueError):
        wrapper.load(nofile)


@pytest.fixture
def load_mgf():
    def loader(file='test.mgf'):
        ctx = MockContext(Config())
        reader = MGFReader(ctx)
        reader.load(os.path.join(fixtures_dir, file))
        return reader
    return loader


def test_duplicated_spectrum(load_mgf):
    reader = load_mgf('duplicated_spectrum.mgf')
    spectra = list(reader.spectra)

    # check if all spectra got created
    assert len(spectra) == 2


def test_set_precursor(load_mgf):
    reader = load_mgf('bigger_spectrum.mgf')
    spectra = list(reader.spectra)

    prev_mass = spectra[0].precursor_mass
    new_mz = (spectra[0].precursor_mz - const.PROTON_MASS) * 2 + const.PROTON_MASS
    spectra[0].precursor_mz = new_mz
    new_mass = spectra[0].precursor_mass
    assert new_mass == prev_mass * 2
    spectra[0].precursor_int = 200
    assert spectra[0].precursor_int == 200
    prev_charge = spectra[0].precursor_charge
    spectra[0].precursor_charge = prev_charge + 1
    assert spectra[0].precursor_charge == prev_charge + 1
    assert spectra[0].precursor_mass == new_mass / prev_charge * (prev_charge + 1)


def test_load_mgf(load_mgf):
    reader = load_mgf()
    expected_precursors = [{'mz': 983.6124, 'charge': 2, 'intensity': 1622453.2378},
                           {'mz': 235.6236, 'charge': 3, 'intensity': 3562467.2343}]
    expected_mz_arrays = [(846.60, 846.80, 847.60), (345.10, 370.20, 460.20)]
    expected_int_arrays = [(73, 44, 67), (237, 128, 108)]
    expected_scan_ids = ['TestFileName controllerType=0 controllerNumber=1 scan=1',
                         'TestFileName controllerType=0 controllerNumber=1 scan=2']
    expected_retention_times = [60, 123]
    expected_file_names = ['test.mgf', 'test.mgf']
    expected_run_names = ['TestFileName', 'TestFileName']
    expected_scan_numbers = [1, 2]

    for i, spectrum in enumerate(reader.spectra):
        assert spectrum.precursor['mz'] == expected_precursors[i]['mz']
        assert spectrum.precursor['charge'] == expected_precursors[i]['charge']
        assert spectrum.precursor['intensity'] == expected_precursors[i]['intensity']
        assert_equal(spectrum.mz_values, expected_mz_arrays[i])
        assert_equal(spectrum.int_values, expected_int_arrays[i])
        assert spectrum.scan_id == expected_scan_ids[i]
        assert spectrum.rt == expected_retention_times[i]
        assert spectrum.file_name == expected_file_names[i]
        assert spectrum.run_name == expected_run_names[i]
        assert spectrum.scan_number == expected_scan_numbers[i]
        assert spectrum.scan_index == i


@pytest.fixture
def load_mzml():
    def loader(file):
        ctx = MockContext(Config())
        reader = MZMLReader(ctx)
        reader.load(os.path.join(fixtures_dir, file))
        return reader
    return loader


@pytest.fixture
def load_raw():
    def loader(file):
        ctx = MockContext(Config())
        reader = RAWReader(ctx)
        reader.file_name = os.path.join(fixtures_dir, file)
        reader.load(reader.file_name)
        return reader
    return loader


test_tiny_mzml_expected = {
    'precursors': [{'mz': 445.33999999999997, 'charge': 2, 'intensity': 120053}],
    'mz_arrays': [(0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)],
    'int_arrays': [(20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0)],
    'scan_ids': ['scan=20'],
    'retention_times': [5.9904999999999999 * 60],
    'file_names': ['tiny.pwiz.1.1.mzML'],
    'run_names': ['tiny1.yep'],
    'scan_numbers': [20],
    'scan_indices': [1],
}

test_orbitrap_raw_expected = {
    'precursor': {'mz': 929.9882, 'charge': 3, 'intensity': 6784054.5},
    'mz_array':
    [120.08098602294921875, 129.32792663574218750, 134.27951049804687500,
     135.89779663085937500, 138.15657043457031250, 141.10256958007812500,
     148.07269287109375000, 148.08961486816406250, 148.10025024414062500,
     148.10411071777343750, 148.11082458496093750, 148.11872863769531250,
     148.12261962890625000, 148.12600708007812500, 148.12963867187500000,
     148.13331604003906250, 148.13685607910156250, 148.14045715332031250,
     148.14399719238281250, 149.06405639648437500, 157.17579650878906250,
     158.09271240234375000, 163.01181030273437500, 168.76603698730468750,
     174.33499145507812500, 175.11894226074218750, 178.26708984375000000,
     178.28367614746093750, 178.29183959960937500, 185.16474914550781250,
     187.14483642578125000, 212.10276794433593750, 215.13945007324218750,
     217.13726806640625000, 228.13433837890625000, 229.11857604980468750,
     230.07667541503906250, 230.54641723632812500, 241.05702209472656250,
     243.13389587402343750, 245.12400817871093750, 245.13220214843750000,
     252.10897827148437500, 253.19094848632812500, 283.10366821289062500,
     284.08731079101562500, 344.44335937500000000, 345.09017944335937500,
     345.10528564453125000, 359.16854858398437500, 397.69671630859375000,
     413.25222778320312500, 424.86209106445312500, 444.40103149414062500,
     446.73138427734375000, 451.23129272460937500, 458.18960571289062500,
     473.21426391601562500, 480.22152709960937500, 514.34307861328125000,
     520.25329589843750000, 526.33514404296875000, 528.48217773437500000,
     548.24493408203125000, 565.27374267578125000, 589.23126220703125000,
     590.32598876953125000, 601.34985351562500000, 601.85302734375000000,
     613.05944824218750000, 627.42449951171875000, 632.03643798828125000,
     657.89245605468750000, 678.35852050781250000, 679.36474609375000000,
     714.43438720703125000, 714.93621826171875000, 715.43756103515625000,
     758.41723632812500000, 759.42163085937500000, 779.36968994140625000,
     779.95477294921875000, 780.45611572265625000, 780.95660400390625000,
     793.38537597656250000, 794.39044189453125000, 836.49725341796875000,
     836.99804687500000000, 837.50213623046875000, 838.00146484375000000,
     845.45050048828125000, 864.45629882812500000, 890.98107910156250000,
     892.45465087890625000, 893.45806884765625000, 895.01007080078125000,
     899.57086181640625000, 932.48260498046875000, 933.48297119140625000,
     936.32177734375000000, 951.03106689453125000, 951.42510986328125000,
     952.03851318359375000, 953.49365234375000000, 983.82409667968750000,
     990.61602783203125000, 991.51928710937500000, 1000.53753662109375000,
     1005.53613281250000000, 1006.54455566406250000, 1008.54846191406250000,
     1009.04632568359375000, 1009.54681396484375000, 1010.04718017578125000,
     1045.56481933593750000, 1046.57043457031250000, 1047.57360839843750000,
     1054.57482910156250000, 1077.55395507812500000, 1101.56872558593750000,
     1142.91845703125000000, 1201.69287109375000000, 1202.69519042968750000,
     1315.77722167968750000, 1350.72644042968750000, 1351.21984863281250000,
     1427.85888671875000000, 1428.73962402343750000, 1598.13989257812500000],
    'int_array':
    [14130.44726562500000000, 4452.05322265625000000, 5412.90380859375000000,
     6305.67138671875000000, 4572.64306640625000000, 8131.85742187500000000,
     6223.08935546875000000, 6725.40429687500000000, 13198.36132812500000000,
     13310.10742187500000000, 16882.59960937500000000, 39185.68359375000000000,
     23673.02539062500000000, 7219.21435546875000000, 12673.29980468750000000,
     13510.87597656250000000, 9292.90917968750000000, 7242.38769531250000000,
     8852.44433593750000000, 4831.40869140625000000, 4886.00585937500000000,
     7632.18847656250000000, 6053.31933593750000000, 5704.05175781250000000,
     5248.47802734375000000, 12812.31542968750000000, 8378.87597656250000000,
     27208.17968750000000000, 7568.13671875000000000, 6338.95458984375000000,
     7476.67724609375000000, 18299.37109375000000000, 7915.58740234375000000,
     7959.14013671875000000, 7163.22607421875000000, 8298.91210937500000000,
     7262.92529296875000000, 6060.22802734375000000, 5558.40917968750000000,
     7015.04199218750000000, 6884.04833984375000000, 8480.23632812500000000,
     7419.19287109375000000, 9165.45703125000000000, 8930.69531250000000000,
     7768.27246093750000000, 7261.61572265625000000, 7287.03320312500000000,
     26536.52929687500000000, 13116.36621093750000000, 6863.51416015625000000,
     8927.03027343750000000, 6449.82910156250000000, 7894.10009765625000000,
     7655.28662109375000000, 14029.04785156250000000, 10296.40234375000000000,
     7210.23437500000000000, 14560.94433593750000000, 8331.16503906250000000,
     9370.45800781250000000, 9671.60546875000000000, 6905.49365234375000000,
     8671.63964843750000000, 27396.90820312500000000, 8466.15039062500000000,
     16554.56445312500000000, 8977.85742187500000000, 9593.01855468750000000,
     7051.41455078125000000, 8936.03710937500000000, 8024.58398437500000000,
     17240.37890625000000000, 23468.41210937500000000, 7809.18798828125000000,
     33339.48828125000000000, 13387.97363281250000000, 10983.28710937500000000,
     32163.31640625000000000, 19095.83789062500000000, 19389.01757812500000000,
     43298.60156250000000000, 59377.23437500000000000, 20826.52734375000000000,
     41160.16796875000000000, 13029.54980468750000000, 22602.69335937500000000,
     14611.32617187500000000, 8125.05517578125000000, 8643.76367187500000000,
     22901.70507812500000000, 17926.45507812500000000, 7408.56494140625000000,
     59861.26562500000000000, 15983.79492187500000000, 8514.04199218750000000,
     7456.01269531250000000, 47363.67578125000000000, 15620.70214843750000000,
     6867.18505859375000000, 19182.57812500000000000, 7642.95947265625000000,
     9118.81445312500000000, 9713.39257812500000000, 8073.67968750000000000,
     7825.13085937500000000, 9543.79394531250000000, 9458.27929687500000000,
     14218.28710937500000000, 10204.81347656250000000, 22068.62500000000000000,
     19712.37890625000000000, 13022.83203125000000000, 10570.73144531250000000,
     44795.09375000000000000, 16410.91406250000000000, 8803.69921875000000000,
     9341.70898437500000000, 9030.94531250000000000, 11737.79101562500000000,
     11414.11230468750000000, 22274.53515625000000000, 27028.65820312500000000,
     8899.01757812500000000, 8322.90625000000000000, 7664.90820312500000000,
     9647.64941406250000000, 7791.89355468750000000, 7994.51660156250000000],
    'scan_id': 103,
    'retention_time': 100.41414908668334 * 60,
    'file_name': 'MS2_MS1_orbitrap.raw',
    'run_name': 'MS2_MS1_orbitrap',
    'scan_number': 103,
    'scan_index': 102
}


def test_load_mzml_tiny(load_mzml):
    reader = load_mzml('tiny.pwiz.1.1.mzML')
    # get all MS2 spectra
    spectra = list(reader.spectra)
    # There should be 4 spectra in total
    assert reader.count_spectra() == 4
    # ... and 1 MS2 spectrum
    assert len(spectra) == 1
    # check spectra values
    check_spectra_reader(reader, test_tiny_mzml_expected)


def test_load_raw(load_raw):
    reader = load_raw('MS2_MS1_orbitrap.raw')
    # get all MS2 spectra (centroid)
    # There should be 68 spectra in total as count_spectra counts only MS2
    assert reader.count_spectra() == 68
    # same for spectra in total 68
    spectra = list(reader.spectra)
    assert len(spectra) == 68
    # check spectra values (scan 103, centroid scan)
    check_spectra_reader_raw(103, reader, test_orbitrap_raw_expected)


def test_load_mzml_small(load_mzml):
    reader = load_mzml('small_zlib.pwiz.1.1.mzML')
    expected_file_name = 'small_zlib.pwiz.1.1.mzML'
    expected_run_name = 'small.RAW'
    # expected values for the first 10 MS2 spectra
    expected_precursors = [
        {'mz': 810.78999999999996, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 837.34000000000003, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 725.36000000000001, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 558.87, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 812.33000000000004, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 810.75, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 837.96000000000004, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 644.05999999999995, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 725.23000000000002, 'charge': np.nan, 'intensity': np.nan},
        {'mz': 559.19000000000005, 'charge': np.nan, 'intensity': np.nan}
    ]

    expected_scan_ids = [
        'controllerType=0 controllerNumber=1 scan=3',
        'controllerType=0 controllerNumber=1 scan=4',
        'controllerType=0 controllerNumber=1 scan=5',
        'controllerType=0 controllerNumber=1 scan=6',
        'controllerType=0 controllerNumber=1 scan=7',
        'controllerType=0 controllerNumber=1 scan=10',
        'controllerType=0 controllerNumber=1 scan=11',
        'controllerType=0 controllerNumber=1 scan=12',
        'controllerType=0 controllerNumber=1 scan=13',
        'controllerType=0 controllerNumber=1 scan=14'
    ]
    expected_retention_times = [
        0.011218333333333334 * 60,
        0.022838333333333332 * 60,
        0.034924999999999998 * 60,
        0.048619999999999997 * 60,
        0.061923333333333337 * 60,
        0.081203333333333336 * 60,
        0.092903333333333324 * 60,
        0.10480333333333333 * 60,
        0.11721500000000001 * 60,
        0.13002166666666667 * 60,
    ]
    expected_scan_numbers = np.array([3, 4, 5, 6, 7, 10, 11, 12, 13, 14])
    expected_scan_indices = expected_scan_numbers - 1

    # get all MS2 spectra
    spectra = list(reader.spectra)

    # There should be 48 spectra in total
    assert reader.count_spectra() == 48
    # ... and 34 MS2 spectrum
    assert len(spectra) == 34
    # check spectra first 10 MS2 spectra values
    for i, spectrum in enumerate(spectra[:10]):
        assert spectrum.precursor['mz'] == expected_precursors[i]['mz']
        assert_allclose(spectrum.precursor['charge'], expected_precursors[i]['charge'],
                        equal_nan=True, atol=0)
        assert_allclose(spectrum.precursor['intensity'], expected_precursors[i]['intensity'],
                        equal_nan=True, atol=0)
        assert spectrum.scan_id == expected_scan_ids[i]
        assert spectrum.rt == expected_retention_times[i]
        assert spectrum.file_name == expected_file_name
        assert spectrum.run_name == expected_run_name
        assert spectrum.scan_number == expected_scan_numbers[i]
        assert spectrum.scan_index == expected_scan_indices[i]


def test_peaklist_wrapper_steps(tmpdir):
    """
    tests if the step and offset arguments work as expected
    """

    ctx = MockContext(Config(crosslinker=[Crosslinker.BS3], ms2_tol='1ppm'))

    pep1 = b'AKT'
    pep2 = b'KLKY'
    pep3 = b'TSKR'
    charge = 2

    ctx.setup_peptide_db(np.array([pep1, pep2, pep3]))
    # create synthetic spectrum MGF
    test_mgf_filepath = os.path.join(tmpdir, 'test.mgf')
    spectra_input = []

    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=0, pep2_index=-1, link_pos1=-1, link_pos2=-1,
        charge=charge, crosslinker=None, context=ctx, isotopes=0))
    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=1, pep2_index=-1, link_pos1=-1, link_pos2=-1,
        charge=charge, crosslinker=None, context=ctx, isotopes=0))
    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=2, pep2_index=-1, link_pos1=-1, link_pos2=-1,
        charge=charge, crosslinker=None, context=ctx, isotopes=0))

    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=0, pep2_index=1, link_pos1=1, link_pos2=0,
        charge=charge, crosslinker=ctx.config.crosslinker[0], context=ctx, isotopes=0))

    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=1, pep2_index=1, link_pos1=0, link_pos2=2,
        charge=charge, crosslinker=ctx.config.crosslinker[0], context=ctx, isotopes=0))

    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=0, pep2_index=2, link_pos1=1, link_pos2=2,
        charge=charge, crosslinker=ctx.config.crosslinker[0], context=ctx, isotopes=0))

    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=1, pep2_index=2, link_pos1=0, link_pos2=2,
        charge=charge, crosslinker=ctx.config.crosslinker[0], context=ctx, isotopes=0))

    spectra_input.append(synthetic_spectra.create_synthetic_spectrum(
        pep1_index=1, pep2_index=2, link_pos1=0, link_pos2=2,
        charge=charge, crosslinker=ctx.config.crosslinker[0], context=ctx, isotopes=0))

    synthetic_spectra.create_synthetic_spectra_mgf(spectra_input, test_mgf_filepath)

    ctx.cluster["all_tasks"] = 2
    ctx.cluster["this_tasks"] = 0
    wrapper1_of_2 = PeakListWrapper(ctx)
    ctx.cluster["this_tasks"] = 1
    wrapper2_of_2 = PeakListWrapper(ctx)
    wrapper1_of_2.load(test_mgf_filepath)
    wrapper2_of_2.load(test_mgf_filepath)

    assert wrapper1_of_2.count_spectra() == wrapper2_of_2.count_spectra() == 4

    ctx.cluster["all_tasks"] = 3
    ctx.cluster["this_task"] = 0
    wrapper1_of_3 = PeakListWrapper(ctx)
    ctx.cluster["this_task"] = 1
    wrapper2_of_3 = PeakListWrapper(ctx)
    ctx.cluster["this_task"] = 2
    wrapper3_of_3 = PeakListWrapper(ctx)
    wrapper1_of_3.load(test_mgf_filepath)
    wrapper2_of_3.load(test_mgf_filepath)
    wrapper3_of_3.load(test_mgf_filepath)
    assert wrapper1_of_3.count_spectra() == 3
    assert wrapper2_of_3.count_spectra() == 3
    assert wrapper3_of_3.count_spectra() == 2
    spectra1 = list(wrapper1_of_3.spectra)
    spectra2 = list(wrapper2_of_3.spectra)
    spectra3 = list(wrapper3_of_3.spectra)

    assert spectra1[0].title == spectra_input[0]['params']['TITLE']
    assert spectra1[1].title == spectra_input[3]['params']['TITLE']
    assert spectra1[2].title == spectra_input[6]['params']['TITLE']

    assert spectra2[0].title == spectra_input[1]['params']['TITLE']
    assert spectra2[1].title == spectra_input[4]['params']['TITLE']
    assert spectra2[2].title == spectra_input[7]['params']['TITLE']

    assert spectra3[0].title == spectra_input[2]['params']['TITLE']
    assert spectra3[1].title == spectra_input[5]['params']['TITLE']

    assert spectra1[0].precursor_mz == spectra_input[0]['params']['PEPMASS']
    assert spectra1[1].precursor_mz == spectra_input[3]['params']['PEPMASS']
    assert spectra1[2].precursor_mz == spectra_input[6]['params']['PEPMASS']

    assert spectra2[0].precursor_mz == spectra_input[1]['params']['PEPMASS']
    assert spectra2[1].precursor_mz == spectra_input[4]['params']['PEPMASS']
    assert spectra2[2].precursor_mz == spectra_input[7]['params']['PEPMASS']

    assert spectra3[0].precursor_mz == spectra_input[2]['params']['PEPMASS']
    assert spectra3[1].precursor_mz == spectra_input[5]['params']['PEPMASS']

    #  make sure this also works when we load additional spectra i.e. via load(xxx, reset=False)
    # This is being used, for example, when reading all files from a directory
    wrapper1_of_3.load(test_mgf_filepath)
    wrapper2_of_3.load(test_mgf_filepath)
    wrapper3_of_3.load(test_mgf_filepath)
    wrapper1_of_3.load(test_mgf_filepath, reset=False)
    wrapper2_of_3.load(test_mgf_filepath, reset=False)
    wrapper3_of_3.load(test_mgf_filepath, reset=False)
    assert wrapper1_of_3.count_spectra() == 6
    assert wrapper2_of_3.count_spectra() == 6
    assert wrapper3_of_3.count_spectra() == 4

    spectra1 = list(wrapper1_of_3.spectra)
    spectra2 = list(wrapper2_of_3.spectra)
    spectra3 = list(wrapper3_of_3.spectra)
    assert len(spectra1) == 6
    assert len(spectra2) == 6
    assert len(spectra3) == 4


def test_peaklist_wrapper_steps_mzml():
    # expected values for the first 10 MS2 spectra
    ctx = MockContext(Config())
    pwall = PeakListWrapper(context=ctx)
    pw = []
    ctx.cluster["all_tasks"] = 3
    ctx.cluster["this_task"] = 0
    pw.append(PeakListWrapper(context=ctx))
    ctx.cluster["this_task"] = 1
    pw.append(PeakListWrapper(context=ctx))
    ctx.cluster["this_task"] = 2
    pw.append(PeakListWrapper(context=ctx))
    mzmlpath = os.path.join(fixtures_dir, 'small_zlib.pwiz.1.1.mzML')
    pwall.load(mzmlpath)
    pw[0].load(mzmlpath)
    pw[1].load(mzmlpath)
    pw[2].load(mzmlpath)

    spectra_count_1 = pw[0].count_spectra()
    spectra_count_2 = pw[1].count_spectra()
    spectra_count_3 = pw[2].count_spectra()
    assert pwall.count_spectra() == spectra_count_1 + spectra_count_2 + spectra_count_3

    spectra = []
    # get all MS2 spectra
    spectra_all = list(pwall.spectra)
    spectra.append(list(pw[0].spectra))
    spectra.append(list(pw[1].spectra))
    spectra.append(list(pw[2].spectra))

    assert len(spectra[0]) == 12
    assert len(spectra[1]) == 11
    assert len(spectra[2]) == 11

    # test for spectra count currently fails, as the count also includes MS1 spectra that are not
    # forwarded
    # assert len(spectra[0]) == spectra_count_1
    # assert len(spectra[1]) == spectra_count_2
    # assert len(spectra[2]) == spectra_count_3

    # check spectra first 10 MS2 spectra values
    for i in range(len(spectra_all)):
        spectrum = spectra[i % 3][int(i/3)]
        assert spectrum.scan_id == spectra_all[i].scan_id


def test_peaklist_wrapper_mixed_formats():
    ctx = MockContext()
    wrapper = PeakListWrapper(ctx)

    # test mixed input file formats
    mzml_file = os.path.join(fixtures_dir, 'tiny.pwiz.1.1.mzML')
    mgf_file = os.path.join(fixtures_dir, 'test.mgf')
    wrapper.load([mzml_file, mgf_file])
    num_spectra = wrapper.count_spectra()
    # tiny.pwiz.1.1.mzML has 4, test.mgf 2 spectra
    assert num_spectra == 6
    assert len(wrapper.readers) == 2
    # check tiny.pwiz.1.1.mzML
    assert wrapper.readers[0].file_name == 'tiny.pwiz.1.1.mzML'
    assert wrapper.readers[0].source_path == mzml_file
    check_spectra_reader(wrapper.readers[0], test_tiny_mzml_expected)
    # check test.mgf
    assert wrapper.readers[1].file_name == 'test.mgf'
    assert wrapper.readers[1].source_path == mgf_file
    check_spectra_reader(wrapper.readers[1], test_mgf_expected)

    # test tar file file containing test.mgf and tiny.pwiz.1.1.mzML
    tar_file = os.path.join(fixtures_dir, 'mgf_mzML_mix.tar.gz')
    wrapper.load(tar_file)
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 6
    assert len(wrapper.readers) == 2
    # check tiny.pwiz.1.1.mzML
    assert wrapper.readers[0].file_name == 'tiny.pwiz.1.1.mzML'
    assert wrapper.readers[0].source_path == tar_file + os.sep + 'tiny.pwiz.1.1.mzML'
    check_spectra_reader(wrapper.readers[0], test_tiny_mzml_expected)
    # check test.mgf
    assert wrapper.readers[1].file_name == 'test.mgf'
    assert wrapper.readers[1].source_path == tar_file + os.sep + 'test.mgf'
    check_spectra_reader(wrapper.readers[1], test_mgf_expected)

    # test zip file containing test.mgf and tiny.pwiz.1.1.mzML
    zip_file = os.path.join(fixtures_dir, 'mgf_mzML_mix.zip')
    wrapper.load(zip_file)
    num_spectra = wrapper.count_spectra()
    assert num_spectra == 6
    assert len(wrapper.readers) == 2
    # check tiny.pwiz.1.1.mzML
    assert wrapper.readers[0].file_name == 'tiny.pwiz.1.1.mzML'
    assert wrapper.readers[0].source_path == zip_file + os.sep + 'tiny.pwiz.1.1.mzML'
    check_spectra_reader(wrapper.readers[0], test_tiny_mzml_expected)
    # check test.mgf
    assert wrapper.readers[1].file_name == 'test.mgf'
    assert wrapper.readers[1].source_path == zip_file + os.sep + 'test.mgf'
    check_spectra_reader(wrapper.readers[1], test_mgf_expected)


def test_annotate_spectrum(tmpdir):
    """
    tests annotate_spectrum to correctly annotates every peak of a synthetic spectrum
    """

    ctx = MockContext(Config(crosslinker=[Crosslinker.BS3], ms2_tol='1ppm',
                             fragmentation={"add_precursor": False},
                             ))

    pep1 = b'AKT'
    pep2 = b'KLKY'
    link_pos1 = 1
    link_pos2 = 0
    charge = 2

    ctx.setup_peptide_db(np.array([pep1, pep2]))
    # create synthetic spectrum MGF
    test_mgf_filepath = os.path.join(tmpdir, 'test.mgf')
    synthetic_spectrum = synthetic_spectra.create_synthetic_spectrum(
        pep1_index=0, pep2_index=1, link_pos1=link_pos1, link_pos2=link_pos2,
        charge=charge, crosslinker=ctx.config.crosslinker[0], context=ctx, isotopes=1)

    synthetic_spectra.create_synthetic_spectra_mgf([synthetic_spectrum], test_mgf_filepath)

    # load MGF
    reader = MGFReader(ctx)
    reader.load(test_mgf_filepath)
    spectrum = next(reader.spectra)

    # detect and reduce isotopes
    detector = IsotopeDetector(ctx)
    detected_spec = detector.process(spectrum)

    # generate fragments
    fragments = fragment_crosslinked_peptide_pair(
        pep1_index=0, pep2_index=1, link_pos1=link_pos1, link_pos2=link_pos2,
        crosslinker=ctx.config.crosslinker[0], context=ctx)
    fragments = spread_charges(fragments, ctx, charge)
    fragments.sort()
    # annotate spectrum
    annotations = detected_spec.annotate_spectrum(fragments, ctx)

    # every isotope reduced peak should have one fragment annotated
    np.testing.assert_almost_equal(annotations['frag_mz'], detected_spec.isotope_cluster_mz_values)
    # peaks should exactly match fragments
    assert all(annotations['abs_error'] == 0)
    assert all(annotations['rel_error'] == 0)
    assert not all(annotations['missing_monoisotopic_peak'])
    assert_array_equal(annotations['peak_mz'], detected_spec.isotope_cluster_mz_values)
    assert_array_equal(annotations['peak_int'], detected_spec.isotope_cluster_intensity_values)

    # all fragments should be annotated (synthetic spectrum)
    same_cols = ['LN', 'loss', 'nlosses', 'stub', 'ion_type', 'term', 'idx', 'pep_id', 'ranges']
    assert_array_equal(annotations[same_cols], fragments[same_cols])
    assert_array_equal(annotations['frag_mz'], fragments['mz'])
    assert_array_equal(annotations['frag_charge'], fragments['charge'])


def test_annotate_spectrum_multiple_matches(tmpdir):
    """
    tests annotate_spectrum with two peptides that throw fragments with almost identical m/z
    """

    ctx = MockContext(Config(crosslinker=[Crosslinker.BS3], ms2_tol='5ppm'))

    pep1 = b'HDRK'
    pep2 = b'KMMK'
    link_pos1 = 3
    link_pos2 = 0
    charge = 1
    ctx.setup_peptide_db(np.array([pep1, pep2]))

    # create synthetic spectrum MGF
    test_mgf_filepath = os.path.join(tmpdir, 'test.mgf')
    synthetic_spectrum = synthetic_spectra.create_synthetic_spectrum(
        pep1_index=0, pep2_index=1, link_pos1=link_pos1, link_pos2=link_pos2,
        charge=charge, crosslinker=ctx.config.crosslinker[0], context=ctx, isotopes=3)
    synthetic_spectra.create_synthetic_spectra_mgf([synthetic_spectrum], test_mgf_filepath)

    # load MGF
    reader = MGFReader(ctx)
    reader.load(test_mgf_filepath)
    spectrum = next(reader.spectra)

    # detect and reduce isotopes
    detector = IsotopeDetector(ctx)
    detected_spec = detector.process(spectrum)
    i_reducer = IsotopeReducer()
    i_reduced = i_reducer.process(detected_spec)
    c_reducer = ChargeReducer(ctx)
    processed_spec = c_reducer.process(i_reduced)
    # generate fragments
    fragments = fragment_crosslinked_peptide_pair(
        pep1_index=0, pep2_index=1, link_pos1=link_pos1, link_pos2=link_pos2,
        crosslinker=ctx.config.crosslinker[0], context=ctx)

    # annotate spectrum
    # The b3 of HDRK (HDR = 409.19424237) and the y3 of KMMK (MMK) (409.19377399) should match the
    # same peaks with a small error ~1ppm. So we should have 2 matches on each peak with
    # ms2_rtol 5e-6
    annotations = processed_spec.annotate_spectrum(fragments, ctx)
    overlap_matches = annotations[np.isclose(annotations['peak_mz'], 409.19377399, rtol=5e-6)]
    assert overlap_matches.size == 2
    overlap_matches = annotations[np.isclose(annotations['peak_mz'], 409.19424237, rtol=5e-6)]
    assert overlap_matches.size == 2

    # and only 1 (b3 of HDRK) with a stricter ms2_rtol of 5e-7
    ctx = MockContext(Config(crosslinker=[Crosslinker.BS3], ms2_tol='0.5ppm'))
    annotations = processed_spec.annotate_spectrum(fragments, ctx)
    overlap_matches = annotations[np.isclose(annotations['peak_mz'], 409.19377399, rtol=5e-7)]
    assert overlap_matches.size == 1
    # and only 1 (y3 of KMMK) with a stricter ms2_rtol of 5e-7
    overlap_matches = annotations[np.isclose(annotations['peak_mz'], 409.19424237, rtol=5e-7)]
    assert overlap_matches.size == 1


def test_order_peptide_ids():
    # create test annotations
    mock_annotations_dtypes = [('frag_charge', np.uint8), ('loss', '<U16'), ('nlosses', np.uint8),
                               ('term', '<U1'), ('idx', np.uint8), ('pep_id', np.uint8)]

    test_annotations = [
        # pep_id 1:
        (1, 'H2O', 1, 'n', 1, 1),  # loss
        (1, 'H20', 1, 'c', 1, 1),  # loss
        (1, 'H20', 1, 'c', 2, 1),  # loss
        # pep_id 2:
        (1, '',    0, 'n', 1, 2),  # primary
        (1, '',    0, 'c', 1, 2),  # primary
        # pep_id 3:
        (1, '',    0, 'n', 1, 3),  # primary
    ]

    test_annotations = np.array(test_annotations, dtype=mock_annotations_dtypes)

    original_indices = order_peptide_ids(test_annotations, n_peps=3)

    # indices should be swapped
    np.testing.assert_array_equal(original_indices, np.array([2, 3, 1]))

    expected_annotations = np.array([
        # old pep_id 1:
        (1, 'H2O', 1, 'n', 1, 3),  # loss
        (1, 'H20', 1, 'c', 1, 3),  # loss
        (1, 'H20', 1, 'c', 2, 3),  # loss
        # old pep_id 2:
        (1, '',    0, 'n', 1, 1),  # primary
        (1, '',    0, 'c', 1, 1),  # primary
        # old pep_id 3:
        (1, '',    0, 'n', 1, 2),  # primary
    ], dtype=mock_annotations_dtypes)

    assert_array_equal(test_annotations, expected_annotations)


def test_annotate_missing_monoisotopic_peaks():
    ctx = MockContext(Config(ms2_tol='10ppm'))
    spectrum = Spectrum(
        {'mz': 200, 'charge': 2},
        [
            1300 + const.C12C13_MASS_DIFF,
            2000.5,   # first isotope peak charge 2 with small error
            2020 + const.C12C13_MASS_DIFF / 3,   # first isotope peak charge 3
            2100 + const.C12C13_MASS_DIFF,
            2125 + const.C12C13_MASS_DIFF / 2,   # first isotope peak charge 2
            2125 + const.C12C13_MASS_DIFF,       # second isotope peak charge 2
            2200,
            2300 + const.C12C13_MASS_DIFF,
            2400,
            2400 + const.C12C13_MASS_DIFF,
            2500 + const.C12C13_MASS_DIFF,    # first isotope peak charge 1
            2600,
            2700,
        ],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        'test_scan')
    spectrum.isotope_cluster_mz_values = spectrum.mz_values
    spectrum.isotope_cluster_intensity_values = spectrum.int_values
    spectrum.isotope_cluster_charge_values = np.array([2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 1])
    fragments = np.array(
        # mz, charge, LN, loss, nlosses, stub, ion_type, term, idx, pep_id, ranges
        [
            # NO match missing monoisotopic with peak 1301 (charge 1) because < 2000 Da
            (1300, 1, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # missing monoisotopic match with peak 2000.5 (charge 2)
            (2000, 2, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # missing monoisotopic match with peak 2020.33 (charge undef)
            (2020, 3, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # missing monoisotopic match with peak 2101 (charge undef)
            (2100, 1, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # missing monoisotopic match  with peak 2125.5 (charge 2)
            (2125, 2, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # NO missing monoisotopic match with peak 2200 (peak has a primary match)
            (2200-const.C12C13_MASS_DIFF, 1, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # primary match with peak 2200 (charge 1)
            (2200, 1, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # missing monoisotopic match with peak 2301 (charge 1)
            (2300, 1, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # missing monoisotopic match with peak 2400 (peak only has loss match)
            (2400-const.C12C13_MASS_DIFF, 1, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # Loss match with peak 2400 (charge 1)
            (2400, 1, True, 'H2O', 1, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # NO match with peak 2501 (charge 1) because charge state doesn't fit
            (2500, 2, True, '', 0, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # NO missing monoisotopic match with peak 2700
            # (peak has a loss match BUT this is also loss fragment)
            (2700-const.C12C13_MASS_DIFF, 1, True, 'H2O', 1, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
            # Loss match with peak 2700 (charge 1)
            (2700, 1, True, 'H2O', 1, '', 'b', 'n', 1, 1, ((0, 0), (0, 0))),
        ],
        dtype=dtypes.fragments)
    annotations = spectrum.annotate_spectrum(fragments, ctx)
    assert annotations.size == 9
    assert_array_equal(annotations['missing_monoisotopic_peak'],
                       [True, True, True, True, False, True, True, False, False])
    assert_array_equal(
        annotations['frag_mz'],
        [2000, 2020, 2100, 2125, 2200, 2300, 2400-const.C12C13_MASS_DIFF, 2400, 2700]
    )

    # errors for missing monoisotopic should be calculated from observed peak to M+1 fragment peak
    m1_fragment_peak = 2000 + const.C12C13_MASS_DIFF/2
    expected_abs_error = 2000.5 - m1_fragment_peak
    assert annotations['abs_error'][0] == expected_abs_error
    assert annotations['rel_error'][0] == expected_abs_error / m1_fragment_peak

    # rest of the errors should be 0
    assert all(annotations['abs_error'][1:] == 0)
    assert all(annotations['rel_error'][1:] == 0)

    # test without match missing monoisotopic
    ctx = MockContext(
        Config(ms2_tol='10ppm', fragmentation={"match_missing_monoisotopic": False}))
    annotations = spectrum.annotate_spectrum(fragments, ctx)
    assert annotations.size == 3
    assert_array_equal(annotations['missing_monoisotopic_peak'], [False, False, False])
    assert_array_equal(annotations['frag_mz'], [2200, 2400, 2700])
    assert all(annotations['abs_error'] == 0)
    assert all(annotations['rel_error'] == 0)


def test_annotate_unique_fragment_annotation():
    """
    Test that a fragment-charge combination is only annotate once (lowest ppm error).
    """
    ctx = MockContext(Config(ms2_tol='10ppm'))
    spectrum = Spectrum(
        {'mz': 200, 'charge': 3},
        [
            318.12676522,  # z=3 y8 (approx. -1ppm)
            318.12803773,  # z=3 y8 (approx. +2ppm)
            318.12899212,  # z=3 y8 (approx. +5ppm)
            333.12345678,  # not-annotated peak
            380.14875375,  # z=3 y10
            412.16616745,  # z=2 y7
            413.16949225,  # z=3 y11 (approx. -5ppm)
            413.17279790,  # z=3 y11 (approx. +3ppm)
            420.98765432,  # not-annotated peak
        ],
        [100, 100, 100, 100, 100, 100, 100, 100, 100],
        'test_scan')
    spectrum.isotope_cluster_mz_values = spectrum.mz_values
    spectrum.isotope_cluster_intensity_values = spectrum.int_values
    spectrum.isotope_cluster_charge_values = np.array([3, 3, 3, 0, 3, 2, 3, 3, 3])
    fragments = np.array(
        # mz, charge, LN, loss, nlosses, stub, ion_type, term, idx, pep_id, ranges
        [
            (318.12708335, 3, True, b'', 0, b'', b'y', b'c', 8, 1, [[5, 13], [0, 0]]),
            (380.14875375, 3, True, b'', 0, b'', b'y', b'c', 10, 1, [[3, 13], [0, 0]]),
            (412.16616745, 2, True, b'', 0, b'', b'y', b'c', 7, 1, [[6, 13], [0, 0]]),
            (413.17155839, 3, True, b'', 0, b'', b'y', b'c', 11, 1, [[2, 13], [0, 0]]),
        ],
        dtype=dtypes.fragments)
    annotations = spectrum.annotate_spectrum(fragments, ctx)
    # check that we only have 4 annotations (and not 7)
    assert annotations.size == 4

    # check we have the correct annotations
    expected_annotations = np.array([
        (0, 3, 318.12708335, 3, True, b'', 0, b'', b'y', b'c', 8, 1, [[5, 13], [0, 0]],
         318.12676522, 100, -1e-06, -0.00031813, False, 0),
        (4, 3, 380.14875375, 3, True, b'', 0, b'', b'y', b'c', 10, 1, [[3, 13], [0, 0]],
         380.14875375, 100, 0, 0, False, 1),
        (5, 2, 412.16616745, 2, True, b'', 0, b'', b'y', b'c', 7, 1, [[6, 13], [0, 0]],
         412.16616745, 100, 0, 0, False, 2),
        (7, 3, 413.17155839, 3, True, b'', 0, b'', b'y', b'c', 11, 1, [[2, 13], [0, 0]],
         413.1727979, 100, 3e-6, 0.00123951, False, 3),
    ], dtype=annotations.dtype)

    equal_cols = ['cluster_id', 'cluster_charge', 'frag_mz', 'frag_charge', 'LN', 'loss',
                  'nlosses', 'stub', 'ion_type', 'term', 'idx', 'pep_id', 'ranges', 'peak_mz']
    assert_array_equal(annotations[equal_cols], expected_annotations[equal_cols])
    assert_allclose(expected_annotations['rel_error'], annotations['rel_error'], atol=1e-7)
