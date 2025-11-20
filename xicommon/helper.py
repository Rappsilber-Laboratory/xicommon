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

"""Test helper utilities for xicommon."""
from xicommon.spectra_reader import Spectrum
from xicommon import dtypes
import numpy as np


class SpectrumMock(Spectrum):
    def __init__(self, mz_array, int_array, charge_array, precursor=None, file_name=''):
        """
        A mock implementation of the Spectrum class for testing purposes.
        Parameters:
            mz_array (array-like): The m/z values of the spectrum peaks.
            int_array (array-like): The intensity values of the spectrum peaks.
            charge_array (array-like): The charge states of the spectrum peaks.
            precursor (dict, optional): Information about the precursor ion. Defaults to None.
            file_name (str, optional): The name of the file associated with the spectrum.
                                       Defaults to an empty string.
        """
        mz_array = np.asarray(mz_array)
        int_array = np.asarray(int_array)
        charge_array = np.asarray(charge_array)

        if precursor is None and len(charge_array) > 0:
            precursor = {'charge': np.max(charge_array)}
        Spectrum.__init__(self, precursor, mz_array, int_array, scan_id=0)
        self.isotope_cluster_peaks = np.array([(x, x)
                                               for x in range(len(mz_array))],
                                              dtype=dtypes.peak_cluster)
        self.isotope_cluster_intensity_values = int_array
        self.isotope_cluster_mz_values = mz_array
        self.isotope_cluster_charge_values = charge_array
        self.peak_has_cluster = np.asarray(charge_array, dtype=bool)
        self.file_name = file_name


def create_fasta(sequences, file_path):
    """Create a FASTA file from sequences."""
    file = open(file_path, 'w')
    for i, sequence in enumerate(sequences):
        if isinstance(sequence, bytes):
            sequence = sequence.decode()
        file.write(f'>sp|exampleP{i}|{sequence}\n')
        file.write(f'{sequence}\n')
    file.close()


def compare_numpy(expected, found, cols=None, atol=1e-9, rtol=1.e-8, do_assert=True, do_print=True):
    """
    Compare two numpy structured arrays based on column types.
    Optionally differences can be printed and also the assert can be switched off.

    Non-numeric fields are compared for equality and numeric are compared with the given tolerance.
    All columns in expected need to be in found as well. Columns inf found that are not in
    expected are ignored

    :param expected (ndarray) - expected values
    :param found (ndarray) - found values to be compared against expected
    :param cols (None|List) - if not None a list of columns to be compared
                             if None then all columns in expected are compared.
    :param atol - absolute tolerance used for comparing numeric types
    :param rtol - relative tolerance for comparing numeric types
    :param do_assert - if differences are found raise an assertion.
                        assertions are raised at the end - after all differences where
                        collected/printed
    :param do_print - True: any found difference are printed out; False nothing is printed
    :return (list) of textual representation of differences
    """
    if cols is None:
        cols = expected.dtype.names
    all_err_msg = []
    for name in cols:
        # different sizes
        if len(expected[name]) != len(found[name]):
            msg = f'\n{name} doesnae match,\nexpected\n{expected[name]},\ngot\n{found[name]}'
            all_err_msg.append(msg)
            if do_print:
                print(msg)
        # numeric types
        elif np.issubdtype(found[name].dtype, np.number):
            if not np.allclose(found[name], expected[name], atol=atol, rtol=rtol, equal_nan=True):
                diff_value = (expected[name].astype(np.float64) - found[name].astype(np.float64))
                ppm_value = diff_value / expected[name]*1000000.0
                msg = f'\n{name} doesnae match,\nexpected\n{expected[name]},\ngot\n' \
                      f'{found[name]}\ndifference:\n' \
                      f'{diff_value}' \
                      f'\nppm difference:\n' \
                      f'{ppm_value}'
                all_err_msg.append(msg)
                if do_print:
                    print(msg)
        # generic comparison of non-numeric types
        else:
            if not np.all(found[name] == expected[name]):
                msg = f'\n{name} doesnae match,\nexpected\n{expected[name]},\ngot\n{found[name]}'
                all_err_msg.append(msg)
                if do_print:
                    print(msg)
    if do_assert:
        assert all_err_msg == []
    return all_err_msg
