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

import traceback
from xicommon.cython import fast_lookup
from xicommon import dtypes
from pyteomics import mgf, mzml
from xicommon import const
import numpy as np
import re
import ntpath
import mmap
from abc import ABC, abstractmethod
import zipfile
import zipfile_deflate64 as zipfile64
import tarfile
import io
import os
import sys
from xicommon.cython import isin_set_long
from lxml.etree import XMLSyntaxError
from .xi_logging import log


class Spectrum:
    def __init__(self, precursor, mz_array, int_array, scan_id, rt=np.nan, file_name='',
                 source_path='', run_name='', scan_number=-1, scan_index=-1, title=''):
        """
        Initialise a Spectrum object.

        :param precursor: (dict) Spectrum precursor information as dict.  e.g. {'mz':
            102.234, 'charge': 2, 'intensity': 12654.35}
        :param mz_array: (ndarray, dtype: float64) m/z values of the spectrum peaks
        :param int_array: (ndarray, dtype: float64) intensity values of the spectrum peaks
        :param scan_id: (str) Unique scan identifier
        :param rt: (str) Retention time in seconds (can be a range, e.g 60-62)
        :param file_name: (str) Name of the peaklist file
        :param source_path: (str) Path to the peaklist source
        :param run_name: (str) Name of the MS run
        :param scan_number: (int) Scan number of the spectrum
        :param scan_index: (int) Index of the spectrum in the file
        """
        self.precursor = precursor
        self.scan_id = scan_id
        self.scan_number = scan_number
        self.scan_index = scan_index
        self.rt = rt
        self.file_name = file_name
        self.source_path = source_path
        self.run_name = run_name
        self.title = title
        mz_array = np.asarray(mz_array, dtype=np.float64)
        int_array = np.asarray(int_array, dtype=np.float64)
        # make sure that the m/z values are sorted asc
        sorted_indices = np.argsort(mz_array)
        self.mz_values = mz_array[sorted_indices]
        self.int_values = int_array[sorted_indices]
        self.peak_has_cluster = None
        self._precursor_mass = None
        self.isotope_cluster = None
        self.isotope_cluster_peaks = None
        self.isotope_cluster_mz_values = None
        self.isotope_cluster_intensity_values = None
        self.isotope_cluster_charge_values = None

    @property
    def precursor_charge(self):
        """Get the precursor charge state."""
        return self.precursor['charge']

    @precursor_charge.setter
    def precursor_charge(self, charge):
        self._precursor_mass = None
        self.precursor['charge'] = charge

    @property
    def precursor_mz(self):
        """Get the precursor m/z."""
        return self.precursor['mz']

    @precursor_mz.setter
    def precursor_mz(self, mz):
        self._precursor_mass = None
        self.precursor['mz'] = mz

    @property
    def precursor_int(self):
        """Get the precursor intensity."""
        return self.precursor['intensity']

    @precursor_int.setter
    def precursor_int(self, int):
        self.precursor['intensity'] = int

    @property
    def precursor_mass(self):
        """Return the neutral mass of the precursor."""
        if self._precursor_mass is None:
            self._precursor_mass = (self.precursor['mz'] - const.PROTON_MASS) *\
                self.precursor['charge']
        return self._precursor_mass

    def initialise_isotope_clusters(self):
        self.peak_has_cluster = np.zeros(len(self.mz_values), dtype=bool)

    def _match_mz(self, mz, context):
        """
        Match mz values to the spectrum for a given mass offset.

        Helper function called from `match_mzs`.

        :param mz: (ndarray) mz values to match
        :param context: (Searcher) search context

        :return: matches array as per `fast_lookup`
        """
        # Use fast lookup implementation to find matches
        tolerances = mz * context.get_ms2_rtol(self.source_path) + \
            context.get_ms2_atol(self.source_path)
        limits = np.empty((2, len(mz), 1), np.float64)
        np.subtract(mz, tolerances, out=limits[0, :, 0])
        np.add(mz, tolerances, out=limits[1, :, 0])
        matches = fast_lookup(self.isotope_cluster_mz_values, None, limits, True, False, False)

        return matches

    def match_fragments(self, mz, charge, primary, context):
        """
        Match fragments to the spectrum.

        If `match_missing_monoisotopic` is configured all fragments that have not already been
        matched with their monoisotopic mass (M) to a peak will be tried to match by their first
        isotopic mass (M+1).
        The following restrictions apply for M+1 matching:
            1. The mass of the fragment must be above 2000 Da.
                (At 2000 Da intensity of M and M+1 is similar assuming Averagine).
            2. No fragment will match as M+1 to peaks that have an M match to a primary fragment.
            3. Loss fragments will not match as M+1 to peaks that have any M match.

        :param mz: (ndarray) mz values of fragments to match
        :param charge: (ndarray or int) charges of fragments to match
        :param primary: (ndarray, bool) if fragments are primary fragments
        :param context: (Searcher) search context

        :return: indices of matched fragments, indices of matched clusters, mz values matched,
                 number of direct (M) matches (not including M+1 matches)
        :rtype: (ndarray), (ndarray), (ndarray), int
        """
        # match fragments to peaks
        matches = self._match_mz(mz, context)

        # filter by correct charge state
        if np.ndim(charge) > 0:
            matched_charge = charge[matches['mass_index']]
            matched_cluster_charge = self.isotope_cluster_charge_values[matches['id']]
            matches = matches[(matched_cluster_charge == 0)
                              | (matched_cluster_charge == matched_charge)]

        matched_frag_indices = matches['mass_index']
        matched_cluster_indices = matches['id']
        matched_mzs = mz[matched_frag_indices]
        matched_primary = primary[matched_frag_indices]

        num_direct = len(matches)

        # match unmatched fragments to peaks with missing monoisotopic offset (M+1)
        if context.config.fragmentation.match_missing_monoisotopic:

            unmatched_frag_indices = np.delete(np.arange(len(mz)), matched_frag_indices)
            unmatched_frag_mzs = mz[unmatched_frag_indices]
            unmatched_frag_charges = charge[unmatched_frag_indices] if np.ndim(charge) else charge
            unmatched_primary_mask = primary[unmatched_frag_indices]

            # only look for missing monoisotopic peaks for fragments with mass larger than ~2000 Da
            large_mask = ((unmatched_frag_mzs * unmatched_frag_charges) > 2000)

            large_unmatched_indices = unmatched_frag_indices[large_mask]
            large_unmatched_mzs = unmatched_frag_mzs[large_mask]
            large_unmatched_charges = \
                unmatched_frag_charges[large_mask] if np.ndim(charge) else charge
            large_unmatched_primary_mask = unmatched_primary_mask[large_mask]

            # Calculate first isotopic peak m/z of fragments and match
            extra_mz = large_unmatched_mzs + const.C12C13_MASS_DIFF / large_unmatched_charges

            extra_matches = self._match_mz(extra_mz, context)

            # filter out primary M+1 matches to peaks with primary M annotation
            extra_primary_mask = large_unmatched_primary_mask[extra_matches['mass_index']]
            already_matched_primary_mask = isin_set_long(extra_matches['id'],
                                                         matches[matched_primary]['id'])
            extra_primary_mask = extra_primary_mask & ~already_matched_primary_mask

            # filter out loss M+1 matches to peaks with any M annotation
            extra_loss_mask = ~extra_primary_mask
            already_matched_mask = isin_set_long(extra_matches['id'], matches['id'])
            extra_loss_mask = extra_loss_mask & ~already_matched_mask

            extra_matches = extra_matches[extra_loss_mask | extra_primary_mask]

            if np.ndim(charge) > 0:
                # filter by correct charge state
                extra_matched_charge = large_unmatched_charges[extra_matches['mass_index']]
                extra_matched_cluster_charge = \
                    self.isotope_cluster_charge_values[extra_matches['id']]
                extra_matches = extra_matches[
                    (extra_matched_cluster_charge == 0)
                    | (extra_matched_cluster_charge == extra_matched_charge)]

            matched_frag_indices = np.concatenate([
                matched_frag_indices,
                large_unmatched_indices[extra_matches['mass_index']]])

            matched_cluster_indices = np.concatenate([
                matched_cluster_indices,
                extra_matches['id']])

            matched_mzs = np.concatenate([
                matched_mzs,
                extra_mz[extra_matches['mass_index']]])

        return matched_frag_indices, matched_cluster_indices, matched_mzs, num_direct

    def annotate_spectrum(self, fragments, context):
        """
        Do full annotation of the spectrum for given fragments.

        This function is the entry point to annotate a spectrum, but actually just
        wraps calls to `match_fragments` and `build_annotation_table`.

        :param fragments: (ndarray) numpy array of fragments to match
        :param context: (Searcher) search context

        :return: (ndarray) array of matched fragments
        """
        primary_mask = fragments['nlosses'] == 0
        matched_frag_indices, matched_cluster_indices, matched_mzs, num_direct = \
            self.match_fragments(fragments['mz'], fragments['charge'], primary_mask, context)

        # assign matched fragments and peaks
        matched_cluster_mzs = self.isotope_cluster_mz_values[matched_cluster_indices]

        # create matched missing monoisotopic array
        mmm_array = np.empty(matched_frag_indices.size)
        mmm_array[:num_direct] = False
        mmm_array[num_direct:] = True

        # calculate errors on fragment mz's that were used for matching
        abs_error = matched_cluster_mzs - matched_mzs
        rel_error = abs_error / matched_mzs

        # sort by smallest absolute (non-negative) relative (relative to the m/z) error
        error_sort = np.argsort(np.absolute(rel_error))
        # do np.unique to get the indices of the matches with the smallest error for each fragment
        _, error_min_idx = np.unique(matched_frag_indices[error_sort], return_index=True)

        # sort and subset the other arrays
        matched_frag_indices = matched_frag_indices[error_sort][error_min_idx]
        matched_cluster_mzs = matched_cluster_mzs[error_sort][error_min_idx]
        matched_cluster_indices = matched_cluster_indices[error_sort][error_min_idx]
        abs_error = abs_error[error_sort][error_min_idx]
        rel_error = rel_error[error_sort][error_min_idx]
        mmm_array = mmm_array[error_sort][error_min_idx]

        matched_frags = fragments[matched_frag_indices]
        matched_cluster_ints = self.isotope_cluster_intensity_values[matched_cluster_indices]
        # Allocate empty annotation table
        annotation_table = np.empty(len(matched_frag_indices), dtypes.annotations)
        # populate annotation table
        annotation_table['cluster_id'] = matched_cluster_indices
        annotation_table['cluster_charge'] = self.isotope_cluster_charge_values[
            matched_cluster_indices]
        annotation_table['frag_mz'] = matched_frags['mz']
        annotation_table['frag_charge'] = matched_frags['charge']
        annotation_table['ion_type'] = matched_frags['ion_type']
        annotation_table['LN'] = matched_frags['LN']
        annotation_table['loss'] = matched_frags['loss']
        annotation_table['nlosses'] = matched_frags['nlosses']
        annotation_table['term'] = matched_frags['term']
        annotation_table['idx'] = matched_frags['idx']
        annotation_table['pep_id'] = matched_frags['pep_id']
        annotation_table['stub'] = matched_frags['stub']
        annotation_table['ranges'] = matched_frags['ranges']
        annotation_table['peak_mz'] = matched_cluster_mzs
        annotation_table['peak_int'] = matched_cluster_ints
        annotation_table['abs_error'] = abs_error
        annotation_table['rel_error'] = rel_error
        annotation_table['missing_monoisotopic_peak'] = mmm_array

        # filter by matching charge states
        undefined_charges = self.isotope_cluster_charge_values[matched_cluster_indices] == 0
        frag_charges = fragments['charge'][matched_frag_indices]
        peak_charges = self.isotope_cluster_charge_values[matched_cluster_indices]
        charge_matches = (frag_charges == peak_charges) | undefined_charges
        annotation_table = annotation_table[charge_matches]

        annotation_table.sort(order='frag_mz')

        return annotation_table


class PeakListWrapper:
    """Wrapper holding SpectraReaders."""

    def __init__(self, context):
        """
        Initialise the PeakListWrapper.

        :param context: (Searcher) Search context
        """
        self.readers = None
        self.context = context
        self.offset = context.cluster["this_task"]
        self.step = context.cluster["all_tasks"]

    def load(self, peaklist_files, reset=True):
        """
        Create SpectraReaders from peaklist files.

        Supported file types: MGF and mzML (and tar or zip archives)
        :param peaklist_files: (str | list of str) path(s) to the peak list file(s) or archive(s)
        """
        if not isinstance(peaklist_files, list):
            peaklist_files = [peaklist_files]

        if reset:
            self.readers = []
        for peaklist_file in peaklist_files:
            count_readers = len(self.readers)
            if not os.path.exists(peaklist_file):
                raise ValueError(f"{peaklist_file} does not exist")
            # check if it is a directory
            if os.path.isdir(peaklist_file):
                # try to load any file in that directory
                for filename in os.listdir(peaklist_file):
                    self.load(os.path.join(peaklist_file, filename), reset=False)
            # check for zipfile
            elif zipfile.is_zipfile(peaklist_file):
                try:
                    zip_f = zipfile.ZipFile(peaklist_file)
                except zipfile.BadZipFile:
                    # assume it is a deflate64 compressed zip file
                    zip_f = zipfile64.ZipFile(peaklist_file)

                for member in zip_f.infolist():
                    self._load(zip_f.open(member), member.filename,
                               peaklist_file + os.sep + member.filename)
            # check for tarfile
            elif tarfile.is_tarfile(peaklist_file):
                tar_f = tarfile.open(peaklist_file)
                for member in tar_f.getmembers():
                    self._load(tar_f.extractfile(member), member.name,
                               peaklist_file + os.sep + member.path)
            # else load file
            else:
                self._load(peaklist_file, peaklist_file)
            if reset:
                if count_readers == len(self.readers):
                    raise ValueError(f"{peaklist_file} could not be loaded")

    def _load(self, stream, filename, source_path=None):
        """Check the file type of the stream and load it."""
        filename = ntpath.basename(filename)
        if filename.lower().endswith('.mgf'):
            if isinstance(stream, str):
                stream = open(stream)
            else:
                stream = io.TextIOWrapper(stream)
            self.readers.append(MGFReader(self.context))
        elif filename.lower().endswith('.mzml'):
            self.readers.append(MZMLReader(self.context))
        elif filename.lower().endswith('.raw') and isinstance(stream, str):
            log('Running into RAW file based search')
            self.readers.append(RAWReader(self.context))
        else:
            return
        self.readers[-1].load(stream, source_path=source_path,
                              file_name=filename, step=self.step, offset=self.offset)

    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Total number of spectra in all files.
        """
        return sum([r.count_spectra() for r in self.readers])

    @property
    def spectra(self):
        try:
            for reader in self.readers:
                if self.context.has_recalibration(reader.source_path):
                    for spectrum in reader.spectra:
                        # recalibrate the precursor
                        spectrum.precursor['mz'] = (
                            spectrum.precursor['mz']
                            + self.context.get_ms1_aerror(reader.source_path)
                        ) / (1 + self.context.get_ms1_rerror(reader.source_path))
                        # recalibrate the spectrum
                        spectrum.mz_values = (
                            spectrum.mz_values
                            + self.context.get_ms2_aerror(reader.source_path)
                        ) / (1 + self.context.get_ms2_rerror(reader.source_path))
                        yield spectrum
                else:
                    for spectrum in reader.spectra:
                        # no recalibration - just forward spectra
                        yield spectrum

        except Exception as e:
            traceback.print_exc()
            raise e


class SpectraReader(ABC):
    """Abstract Base Class for all SpectraReader."""

    def __init__(self, context):
        """
        Initialize the SpectraReader.

        :param context: (Searcher) Search context
        """
        self.config = context.config
        self._reader = None
        self._re_scan_number = re.compile(self.config.re_scan_number)
        self._re_run_name = re.compile(self.config.re_run_name)
        self. _source = None
        self.file_name = None
        self.source_path = None
        self.default_run_name = self.file_name

    @abstractmethod
    def load(self, source, file_name=None, source_path=None):
        """
        Load the spectrum file.

        :param source: Spectra file source
        :param file_name: (str) filename
        :param source_path: (str) path to the source file (peak list file or archive)
        """
        self._source = source
        if source_path is None:
            if isinstance(source, str):
                self.source_path = source
            elif issubclass(type(source), io.TextIOBase) or \
                    issubclass(type(source), tarfile.ExFileObject):
                self.source_path = source.name
        else:
            self.source_path = source_path

        if file_name is None:
            self.file_name = ntpath.basename(self.source_path)
        else:
            self.file_name = file_name

    @abstractmethod
    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Number of spectra in the file
        """
        ...

    @property
    @abstractmethod
    def spectra(self):
        """Create a Spectra generator."""
        while False:
            yield None


class MGFReader(SpectraReader):
    """SpectraReader for MGF files."""

    def load(self, source, file_name=None, source_path=None, offset=0, step=1):
        """
        Load MGF file.

        :param source: file source, path or stream
        :param file_name: (str) MGF filename
        :param source_path: (str) path to the source file (MGF or archive)
        """
        self._reader = mgf.read(source, use_index=False)
        self.offset = offset
        self.step = step
        super().load(source, file_name, source_path)

    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Number of spectra in the file.
        """
        if issubclass(type(self._source), io.TextIOBase):
            text = self._source.read()
            result = len(list(re.findall('BEGIN IONS', text)))
            self._source.seek(0)
        else:
            with open(self._source, 'r+') as f:
                text = mmap.mmap(f.fileno(), 0)
                result = len(list(re.findall(b'BEGIN IONS', text)))
                text.close()
        count_from_offset = result-self.offset
        return int(count_from_offset/self.step) + (count_from_offset % self.step > 0)

    def _convert_spectrum(self, scan_index, mgf_spec):
        precursor = {
            'mz': mgf_spec['params']['pepmass'][0],
            'charge': mgf_spec['params']['charge'][0],
            'intensity': mgf_spec['params']['pepmass'][1]
        }

        # use title as scan_id, default to filename_scan_index (very unlikely to not have a
        # title but it's not required)
        scan_id = mgf_spec['params'].get('title', '{}_{}'.format(self.file_name, scan_index))

        # parse retention time, default to NaN
        rt = mgf_spec['params'].get('rtinseconds', np.nan)

        # try to parse scan number and run_name from title
        title = mgf_spec['params'].get('title', '')
        run_name_match = re.search(self._re_run_name, title)
        try:
            run_name = run_name_match.group(1)
        except AttributeError:
            run_name = self.default_run_name

        scan_number_match = re.search(self._re_scan_number, title)
        try:
            scan_number = int(scan_number_match.group(1))
        except (AttributeError, ValueError):
            scan_number = -1

        return Spectrum(precursor, mgf_spec['m/z array'],
                        mgf_spec['intensity array'], scan_id,
                        rt, self.file_name, self.source_path, run_name, scan_number,
                        scan_index, title=title)

    @property
    def spectra(self):
        """Generator wrapped around pyteomics generator. Reformatting the spectrum information."""
        if (self.step == 1 and self.offset == 0):
            # just forward everything
            for scan_index, mgf_spec in enumerate(self._reader):
                yield self._convert_spectrum(scan_index, mgf_spec)
        else:
            # forward only every nth spectrum
            count = -self.offset
            for scan_index, mgf_spec in enumerate(self._reader):
                if count >= 0 and count % self.step == 0:
                    yield self._convert_spectrum(scan_index, mgf_spec)
                count += 1


class MZMLReader(SpectraReader):
    """SpectraReader for mzML files."""

    def load(self, source, file_name=None, source_path=None, offset=0, step=1):
        """
        Read in spectra from an mzML file and stores them as Spectrum objects.

        :param source: file source, path or stream
        :param file_name: (str) mzML filename
        :param source_path: (str) path to the source file (mzML or archive)
        """
        self.offset = offset
        self.step = step
        self._reader = mzml.read(source)
        if self._reader.index is None:
            self._reader = mzml.read(source, use_index=True)
        super().load(source, file_name, source_path)

        # get the default run name
        if issubclass(type(self._source), tarfile.ExFileObject) or \
                issubclass(type(self._source), zipfile.ZipExtFile):
            text = self._source.read()
            result = re.finditer(b'defaultSourceFileRef="(.*)"', text)
            try:
                result = result.__next__().groups()
            except StopIteration:
                result = None
            self._source.seek(0)
        else:
            with open(self._source, 'r+') as f:
                text = mmap.mmap(f.fileno(), 0)
                result = re.finditer(b'defaultSourceFileRef="(.*)"', text)
                try:
                    result = result.__next__().groups()
                except StopIteration:
                    result = None
                text.close()
        if result is not None:
            self.default_run_name = self._reader.get_by_id(result[0].decode('ascii'))['name']
        else:
            # try to get the default run name from sourceFileList:
            try:
                source_files = list(self._reader.iterfind('//sourceFileList'))[0]
                # if there is more than one entry we can't determine the default
                if source_files['count'] != 1:
                    self.default_run_name = file_name
                else:
                    self.default_run_name = source_files['sourceFile'][0]['name']
            except XMLSyntaxError:
                self.default_run_name = file_name
            self.reset()

    def reset(self):
        """Reset the reader."""
        if issubclass(type(self._source), tarfile.ExFileObject) or \
                issubclass(type(self._source), zipfile.ZipExtFile):
            self._source.seek(0)
            self._reader = mzml.read(self._source)
        else:
            self._reader.reset()

    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Number of spectra in the file.
        """
        count_from_offset = len(self._reader) - self.offset
        return int(count_from_offset/self.step) + (count_from_offset % self.step > 0)

    def _convert_spectrum(self, spec):

        # check for single scan per spectrum
        if spec['scanList']['count'] != 1:
            raise ValueError(
                "xiSEARCH2 currently only supports a single scan per spectrum.")
        scan = spec['scanList']['scan'][0]

        # check for single precursor per spectrum
        if spec['precursorList']['count'] != 1 or \
                spec['precursorList']['precursor'][0]['selectedIonList']['count'] != 1:
            raise ValueError(
                "xiSEARCH2 currently only supports a single precursor per spectrum.")
        p = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]

        # create precursor dict
        precursor = {
            'mz': p['selected ion m/z'],
            'charge': p.get('charge state', np.nan),
            'intensity': p.get('peak intensity', np.nan)
        }

        # id is required in mzML so set this as scan_id
        scan_id = spec['id']

        # index is also required in mzML so just use this
        scan_index = spec['index']

        # parse retention time, default to NaN
        rt = scan.get('scan start time', np.nan)
        rt = rt * 60

        # sourceFileRef can optionally reference the 'id' of the appropriate sourceFile.
        if hasattr(spec, 'sourceFileRef'):
            run_name = self._reader.get_element_by_id(spec['sourceFileRef'])['name']
        else:
            run_name = self.default_run_name

        # try to parse scan number from scan_id
        scan_number_match = re.search(self._re_scan_number, scan_id)
        try:
            scan_number = int(scan_number_match.group(1))
        except (AttributeError, ValueError):
            scan_number = None

        return Spectrum(precursor, spec['m/z array'],
                        spec['intensity array'], scan_id,
                        rt, self.file_name, self.source_path, run_name, scan_number, scan_index)

    @property
    def spectra(self):
        """Spectra generator wrapped around pyteomics generator."""

        if (self.step == 1 and self.offset == 0):
            # just forward everything
            for spec in self._reader:
                # skip non-MS2
                if spec['ms level'] != 2:
                    continue
                yield self._convert_spectrum(spec)
        else:
            # forward only every nth spectrum
            count = -self.offset
            for spec in self._reader:
                if spec['ms level'] != 2:
                    continue
                if count >= 0 and count % self.step == 0:
                    yield self._convert_spectrum(spec)
                count += 1


class RAWReader(SpectraReader):
    """SpectraReader for RAW files."""

    def __init__(self, context):
        """
        Initialize the SpectraReader.

        :param context: (Searcher) Search context
        """
        super().__init__(context)
        try:
            from fisher_py.raw_file_reader import RawFileReaderAdapter
            self.RawFileReaderAdapter = RawFileReaderAdapter
            from fisher_py.data.filter_enums import MsOrderType
            self.MsOrderType = MsOrderType
            from fisher_py.data import Device
            self.Device = Device
        except Exception:
            log('RAW file initialisation failed - will not be able to read RAW files')
            sys.exit(1)

    def load(self, source, file_name=None, source_path=None, offset=0, step=1):
        """
        Load RAW file.

        :param source: file source, path or stream
        :param file_name: (str) RAW filename
        :param source_path: (str) path to the source file (RAW or archive)
        """

        self.offset = offset
        self.step = step
        self._reader = self.RawFileReaderAdapter.file_factory(source)
        self.instrument_index = self._reader.instrument_count - 1
        self.sample_name = self._reader.sample_information.sample_name
        self.sample_id = self._reader.sample_information.sample_id
        file_name = os.path.basename(source) if file_name is None else file_name
        self.default_run_name = os.path.splitext(os.path.basename(file_name))[0]
        self.number_of_ms2 = 0

        # Choose the data stream from the data source.
        self._reader.select_instrument(self.Device.MS, 1)

        log(f'The RAW file has data from {self._reader.instrument_count} instruments')
        super().load(source, file_name, source_path)

    def count_spectra(self):
        """
        Count the number of spectra.
        :return (int) Number of spectra in the file.
        """

        for scan_number in range(1, self._reader.run_header_ex.spectra_count + 1):
            scan_statistics = self._reader.get_scan_stats_for_scan_number(scan_number)
            scan_filter = self._reader.get_filter_for_scan_number(scan_number)

            if (scan_statistics.is_centroid_scan):
                if scan_filter.ms_order == self.MsOrderType.Ms2:
                    self.number_of_ms2 += 1
        count_from_offset = self.number_of_ms2 - self.offset
        return int(count_from_offset/self.step) + (count_from_offset % self.step > 0)

    def find_precursor_intensity(self, raw_file, precursor_mz, ms1_scan_number):
        """
        Finds the intensity of the precursor ion in the MS1 scan.
        :param raw_file: The RawFileAccess object to access raw file data.
        :param precursor_mz: The m/z value of the precursor ion.
        :param ms1_scan_number: The scan number of the MS1 scan (master scan).
        :return: The intensity of the precursor ion in the MS1 scan or NaN if not found.
        """
        try:
            scan_statistics = raw_file.get_scan_stats_for_scan_number(ms1_scan_number)
            if scan_statistics.is_centroid_scan:
                centroid_stream = raw_file.get_centroid_stream(ms1_scan_number, False)
                mz_arr = np.array(getattr(centroid_stream, "masses", []), dtype=float)
                intensities_arr = np.array(getattr(centroid_stream, "intensities", []), dtype=float)
            else:
                segmented_scan = raw_file.get_segmented_scan_from_scan_number(ms1_scan_number,
                                                                              scan_statistics)
                mz_arr = np.array(getattr(segmented_scan, "positions", []), dtype=float)
                intensities_arr = np.array(getattr(segmented_scan, "intensities", []),
                                           dtype=float)

            if mz_arr.size == 0 or intensities_arr.size == 0:
                return np.nan  # No valid scan data available

            idx = np.searchsorted(mz_arr, precursor_mz, side='left')

            # Ensure idx is within values bounds
            if idx >= intensities_arr.size:
                return np.nan

            return intensities_arr[idx]

        except Exception:
            return np.nan

    def _convert_spectrum(self, scan_number=int):
        """Convert spectrum to precursor dictionary."""
        precursor = {
            'mz': np.nan,
            'charge': np.nan,
            'intensity': np.nan
        }

        master_scan = {
            'master_scan_number': np.nan,
            'master_scan_index': np.nan
        }

        intensities_arr = []
        mz_arr = []

        retention_time = self._reader.retention_time_from_scan_number(scan_number)
        rt = retention_time * 60

        scan_id = scan_number
        scan_index = scan_number - 1

        centroid_stream = self._reader.get_centroid_stream(scan_number, False)
        if centroid_stream.length == 0:
            return None
        mz_arr = np.array(getattr(centroid_stream, "masses", []), dtype=np.float64)
        intensities_arr = np.array(getattr(centroid_stream, "intensities", []), dtype=np.float64)

        trailer_data = self._reader.get_trailer_extra_information(scan_number)
        trailer_labels = [x[:-1] if x[-1] == ":" else x for x in trailer_data.labels]
        trailer_dict = dict(zip(trailer_labels, trailer_data.values))

        precursor['mz'] = float(trailer_dict.get('Monoisotopic M/Z', np.nan))
        precursor['charge'] = int(trailer_dict.get('Charge State', -1))

        master_scan['master_scan_number'] = int(trailer_dict.get('Master Scan Number', -1))
        master_scan['master_scan_index'] = int(trailer_dict.get('Master Index', -1))

        precursor['intensity'] = self.find_precursor_intensity(self._reader, precursor['mz'],
                                                               master_scan['master_scan_number'])

        return Spectrum(precursor, mz_arr, intensities_arr, scan_id, rt,
                        self.file_name, self.source_path,
                        self.default_run_name, scan_number, scan_index)

    @property
    def spectra(self):
        """Spectra generator."""
        count = -self.offset
        for scan_number in range(1, self._reader.run_header_ex.spectra_count + 1):
            # Build scan-specific statistics
            scan_statistics = self._reader.get_scan_stats_for_scan_number(scan_number)
            scan_filter = self._reader.get_filter_for_scan_number(scan_number)

            if (scan_statistics.is_centroid_scan):

                if scan_filter.ms_order == self.MsOrderType.Ms2:
                    if count >= 0 and count % self.step == 0:
                        spec = self._convert_spectrum(scan_number)
                        if spec is not None:
                            yield spec
                    count += 1
            else:
                continue  # Case of having MS1 spectra


def order_peptide_ids(annotations, n_peps):
    """
    Reassign order of peptide ids in the annotation table by number of primary fragments (desc).

    :param annotations: (ndarray) Structured numpy array with fragment annotations
    :param n_peps: (int) Number of peptides that were searched
    :return: (list, int) list of original pep_ids in the new order
    """
    # create a copy of the annotations to keep track of the old pep_ids
    old_annotations = np.copy(annotations[['nlosses', 'pep_id']])

    # Add in one extra primary fragment per peptide to circumvent problems for peptides without
    # (primary) annotations
    extra_table = np.zeros(n_peps, dtype=old_annotations.dtype)
    extra_table['pep_id'] = np.arange(1, n_peps+1)
    count_table = np.concatenate([old_annotations, extra_table])

    # get the unique pep_ids and their counts in the primary annotations
    primary = count_table[count_table['nlosses'] == 0]
    unique_pep_ids, count_pep_ids = np.unique(primary['pep_id'], return_counts=True)

    # create a structured numpy array to order pep_ids based on their occurrence
    pep_id_order_arr = np.empty(count_pep_ids.size, dtype=[('count', np.uintp), ('id', np.uint8)])
    pep_id_order_arr['count'] = count_pep_ids
    pep_id_order_arr['id'] = unique_pep_ids
    pep_id_order_arr[::-1].sort(order='count')

    # loop over the ordered peptide ids
    for new_id, old_id in enumerate(np.nditer(pep_id_order_arr['id'])):
        # overwrite the pep_id in the annotations where the old_annotations have the old pep_id
        annotations['pep_id'][old_annotations['pep_id'] == old_id] = new_id + 1

    return pep_id_order_arr['id']
