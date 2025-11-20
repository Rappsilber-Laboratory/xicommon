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

#cython: language_level=3,
# distutils : language = c++

import numpy as np
cimport numpy as np
cimport cython
from xicommon import dtypes, const
from xicommon.mass import aa_masses, ion_mass
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

cdef packed struct crange_dtype:
    np.uint8_t r_from
    np.uint8_t r_to

ctypedef char[25] closs_t

cdef packed struct cfragment_dtype:
    # ('mz', np.float64),             # fragment m/z value
    double  mz
    # ('charge', np.uint8),           # fragment charge state
    np.uint8_t charge
    # ('LN', np.bool_),               # linear (True) or crosslinked fragment (False)
    np.uint8_t LN
    # ('loss', '<U25'),               # name of the neutral loss or '' for primary fragment
    closs_t loss
    # ('nlosses', np.uint8),          # number of times the loss occurred
    np.uint8_t nlosses
    # ('stub', '<U1'),                # cleaved crosslinker stub type
    char stub
    # ('ion_type', '<U1'),            # ion type of the fragment, e.g. 'b', 'y' or 'P'
    char ion_type
    # ('term', '<U1'),                # terminus of the fragment ('n' or 'c')
    char term
    # ('idx', np.uint8),              # fragment number
    np.uint8_t idx
    # ('pep_id', np.uint8),           # which peptide this fragment belongs to
    np.uint8_t pep_id
    # ('ranges', np.uint8, (2, 2))    # range over the two peptides as 2x2 array (start, end)
    np.uint8_t[2][2] ranges


cdef packed struct cfragment_minimal_dtype:
    double mz           # fragment m/z value
    char ion_type       # ion type of the fragment, e.g. 'b', 'y' or 'P'
    char term           # terminus of the fragment ('n' or 'c')
    np.uint8_t idx      # offset into peptide from the terminus
    np.uint8_t start    # start index in the peptide
    np.uint8_t end      # end index in the peptide


cdef class Fragmentation:
    """
    Provides a cython based fragmentation of peptides
    """

    # making an array with space for all basic aminoacids indexed by their ordinal number (ord('Z') = 90)
    cdef double[90] amino_acids_masses
    # view for the modification masses - 0 - no modification; 1 first defined modificiation;...
    cdef double[:] modification_masses
    # names of the terminal ions - as defined in the config
    cdef unsigned char[:] nterm_ion_names
    cdef unsigned char[:] cterm_ion_names
    # masses of the defined terminal ions
    cdef double[:] nterm_ion_masses
    cdef double[:] cterm_ion_masses
    # flag (0 or 1) whther precusor ions should be created
    cdef int include_precursor
    # how many ion types should be generated in total
    cdef int count_ions
    # how many n-terminal ion types should be generated
    cdef int count_nterm_ions
    # how many c-terminal ion types should be generated
    cdef int count_cterm_ions
    # the search context
    cdef context

    # some masses that get copied out from const
    cdef double proton_mass
    cdef double H2O_mass
    cdef double OH_mass
    cdef double H_mass

    # if we look at exactly one n-terminal ion type and one c-terminal ion type we can produce
    # slightly faster code by omitting an inner loop - this is the flag to say "do this"
    cdef int single_ion_mode

    def __init__(self, context):
        self.context = context
        self.set_aminoacid_masses(aa_masses)

        # transfer modification infos from the config as 1 based index
        cdef double[:] modmasses = np.empty(len(context.config.modification.modifications) + 1)
        # zer0 is no modification = zero mass
        modmasses[0] = 0.0
        cdef int mod_id
        for mod_id, mod in enumerate(context.config.modification.modifications):
            modmasses[mod_id + 1] = mod.mass
        self.modification_masses = modmasses

        # transfer some constants
        self.proton_mass = const.PROTON_MASS
        self.H2O_mass = const.H2O_MASS
        self.H_mass = const.H_MASS
        self.OH_mass = const.H2O_MASS - const.H_MASS

        self.nterm_ion_names = np.array([ord(i) for i in context.config.fragmentation.nterm_ions],
                                     dtype=np.uint8)
        self.nterm_ion_masses = np.array(
            [self.H2O_mass + ion_mass(i) for i in context.config.fragmentation.nterm_ions])

        self.cterm_ion_names = np.array([ord(i) for i in context.config.fragmentation.cterm_ions],
                                    dtype=np.uint8)
        self.cterm_ion_masses = np.array(
            [self.H2O_mass + ion_mass(i) for i in context.config.fragmentation.cterm_ions])

        self.count_nterm_ions = self.nterm_ion_masses.shape[0]
        self.count_cterm_ions = self.cterm_ion_masses.shape[0]
        self.count_ions = self.count_nterm_ions + self.count_cterm_ions

        # should we enable the single c and n-terminal ion mode?
        if (self.count_nterm_ions == 1 and self.count_cterm_ions == 1):
            self.single_ion_mode = 1

        if context.config.fragmentation.add_precursor:
            self.include_precursor = 1
        else:
            self.include_precursor = 0

    def __dealloc__(self):
        pass

    def set_aminoacid_masses(self, double[:] aa_masses):
        """
        Define the amino acid masses used for fragmentation.

        Assumes that the amino acids are only defined for chars A-Z and the ASCII code for the
        character is the index  into the array.
        Currently only the entries for indices 65 to 89 are transferred.
        :param aa_masses: (double[:]) the array with amino acid masses
        """
        cdef double[:] masses = aa_masses
        for m in range(0,65):
            self.amino_acids_masses[m] = 0
        for m in range(65,90):
            self.amino_acids_masses[m] = masses[m]


    def fragment_peptide(self, peptide_index, bint add_precursor=False):
        """
        Generate the basic fragments of a peptide for multiple c- and n-ion types.
        This is the python callable wrapper for _fragment

        :param peptide_index - index into the modified peptide database
        :param add_precursor - Whether to include the precursor or not
        :return: fragment array
        """
        peptide = self.context.peptide_db.unmod_pep_sequence(peptide_index)
        peptide_mass = self.context.peptide_db.peptide_mass(peptide_index) + self.proton_mass
        cdef int peptide_length = len(peptide)
        mod_pep = self.context.peptide_db.peptides[peptide_index]
        mod_arr = mod_pep['modifications']
        cdef int precursor_count
        if add_precursor:
            precursor_count = 1
        else:
            precursor_count = 0

        cdef np.ndarray fragments = self._fragment(peptide, mod_arr, peptide_length,
                                                   precursor_count, peptide_mass)

        return fragments

    def fragment_peptide_minimal(self, peptide_index, bint add_precursor=False):
        """
        Generate the basic fragments of a peptide for multiple c- and n-ion types.
        This is the python callable wrapper for _fragment_minimal

        :param peptide_index - index into the modified peptide database
        :param add_precursor - Whether to include the precursor or not
        :return: fragment array
        """

        peptide = self.context.peptide_db.unmod_pep_sequence(peptide_index)
        cdef int peptide_length = len(peptide)
        mod_pep = self.context.peptide_db.peptides[peptide_index]
        mod_arr = mod_pep['modifications']
        cdef int precursor_count
        if add_precursor:
            precursor_count = 1
        else:
            precursor_count = 0
        peptide_mass = self.context.peptide_db.peptide_mass(peptide_index) + self.proton_mass

        cdef np.ndarray fragments = self._fragment_minimal(peptide, mod_arr, peptide_length,
                                                          precursor_count, peptide_mass)

        return fragments

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef inline void _fill_fragment_row(self, cfragment_dtype[:] frags, int  row, int idx,
                                        double single_charge_mz, unsigned char term,
                                        unsigned char iontype, np.uint8_t r_from, np.uint8_t r_to):
        """
        Fill a single row of the fragment-table with the given values.
        
        :param frags - the fragment-table
        :param row - into which row to store fragment
        :param idx - the index of the fragment (e.g. 9 for y9 ion)
        :param single_charge_mz - the expected single charged m/z value for the fragment 
        :param term - b'n' if it is an n-terminal fragment b'c' if cterminal
        :param iontype - ion type of the fragment, e.g. b, y or P
        :param r_from - start of the fragment range
        :param r_to - end of the fragment range
        """
        frags[row].mz = single_charge_mz
        frags[row].charge = 1
        frags[row].LN = 1
        memset(frags[row].loss, 0,  sizeof(closs_t))
        frags[row].nlosses = 0
        frags[row].stub = 0
        frags[row].term = term
        frags[row].idx = idx
        frags[row].ion_type = iontype
        frags[row].pep_id = 1
        frags[row].ranges[0][0] = r_from
        frags[row].ranges[0][1] = r_to
        frags[row].ranges[1][0] = 0
        frags[row].ranges[1][1] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef np.ndarray _fragment(self, unsigned char* base_sequence,
                              np.uint8_t[:] modifications,
                              int peptide_length,
                              int include_precursor,
                              double peptide_mass):
        """
        Generate the basic fragments of a peptide for multiple c- and n-ion types.
        
        :param base_sequence - the unmodified peptide sequence
        :param modifications - modification indexes in format
                            nterm,cterm,res0,res1,res2...
                            0 means no modification
                            1 first modification defined in the config
                            2 second modification defined in the config
                            ...
        :param peptide_length - number of residues in the peptide
        :param include_precursor - Whether to include the precursor or not
        :param peptide_mass - singly charged mass of the peptide to be fragmented
                              only used for precursor ion generation
        :return: fragment array
        """

        cdef int frags_per_ions = (peptide_length - 1)
        # how many fragments need to be generated
        cdef int fragment_count = frags_per_ions * self.count_ions + include_precursor

        # the array itself as we want to return the array not a view of it
        cdef np.ndarray return_frags_array = np.empty(fragment_count, dtype=dtypes.fragments)
        # reserve space for the fragments to be generated
        cdef cfragment_dtype[:] return_frags = return_frags_array

        # array for the masses of each residue
        #cdef double[:] residue_mass = np.empty(peptide_length,np.float64)
        cdef double *residue_mass = <double *> malloc(peptide_length * sizeof(double))

        # initialise the mass for c and nterminal fragments with the c and n-terminal modifications
        cdef double* nmasses = <double *> malloc(self.count_nterm_ions * sizeof(double))
        cdef double* cmasses = <double *> malloc(self.count_cterm_ions * sizeof(double))
        memcpy(nmasses,&self.nterm_ion_masses[0],self.count_nterm_ions * sizeof(double))
        memcpy(cmasses,&self.cterm_ion_masses[0],self.count_cterm_ions * sizeof(double))


        cdef double pepmass
        cdef int i
        for i in range(self.count_nterm_ions):
            nmasses[i] += self.proton_mass
        for i in range(self.count_cterm_ions):
            cmasses[i] += self.proton_mass

        cdef int lastidx = fragment_count - 1

        # translate the residues to masses
        cdef int r
        for r in range(peptide_length):
            residue_mass[r] = self.amino_acids_masses[base_sequence[r]] + \
                              self.modification_masses[modifications[r + 2]]
        # add terminal modification masses
        if modifications[0]:
            residue_mass[0] += self.modification_masses[modifications[0]]
        if modifications[1]:
            residue_mass[peptide_length - 1] += self.modification_masses[modifications[1]]

        # make separate space for one set of c and n terminal fragments - independent of actual
        # fragment type

        cdef int nextrow = 0
        # now iterate over the residues
        cdef int cion
        cdef int nion
        cdef int cres = peptide_length - 1
        if self.single_ion_mode:
            for r in range(peptide_length - 1):
                # next n-terminal mass
                nmasses[0] += residue_mass[r]
                self._fill_fragment_row(return_frags, nextrow, r + 1, nmasses[0], b'n',
                                        self.nterm_ion_names[0], 0, r + 1)
                nextrow += 1

                cmasses[0] += residue_mass[cres]
                self._fill_fragment_row(return_frags, nextrow, r + 1, cmasses[0], b'c',
                                        self.cterm_ion_names[0], cres, peptide_length)
                nextrow += 1
                cres -= 1

        else:
            for r in range(peptide_length - 1):
                # next n-terminal mass
                for nion in range(self.count_nterm_ions):
                    nmasses[nion] += residue_mass[r]
                    self._fill_fragment_row(return_frags, nextrow, r + 1, nmasses[nion], b'n',
                                            self.nterm_ion_names[nion], 0, r + 1)
                    nextrow += 1

                for cion in range(self.count_cterm_ions):
                    cmasses[cion] += residue_mass[cres]
                    self._fill_fragment_row(return_frags, nextrow, r + 1, cmasses[cion], b'c',
                                            self.cterm_ion_names[cion], cres, peptide_length)
                    nextrow += 1
                cres -= 1

        if include_precursor:
            self._fill_fragment_row(return_frags, lastidx, peptide_length,
                                    peptide_mass, b'P', b'P', 0, peptide_length)
        free(residue_mass)
        free(nmasses)
        free(cmasses)
        return return_frags_array


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef np.ndarray _fragment_minimal(self, unsigned char* base_sequence,
                 np.uint8_t[:] modifications, int peptide_length,
                 int include_precursor, double peptide_mass):
        """
        Generate the basic fragments of a peptide for multiple c- and n-ion types.
        Generates a reduced numpy array with fewer information then _fragment(). 
        
        :param base_sequence - the unmodified peptide sequence
        :param modifications - modification indexes in format
                            nterm,cterm,res0,res1,res2...
                            0 means no modification
                            1 first modification defined in the config
                            2 second modification defined in the config
                            ...
        :param peptide_length - number of residues in the peptide
        :param include_precursor - Whether to include the precursor or not
        :param peptide_mass - singly charged mass of the peptide to be fragmented
                              only used for precursor ion generation
        :return: fragment array
        """

        cdef int frags_per_ions = (peptide_length - 1)
        # how many fragments need to be generated
        cdef int fragment_count = frags_per_ions * self.count_ions + include_precursor

        # the array itself as we want to return the array not a view of it
        cdef np.ndarray return_frags_array = np.empty(fragment_count, dtype=dtypes.basic_fragments)
        # reserve space for the fragments to be generated
        cdef cfragment_minimal_dtype[:] return_frags = return_frags_array

        # array for the masses of each residue
        cdef double *residue_mass = <double *> malloc(peptide_length * sizeof(double))

        # initialise the mass for c and nterminal fragments with the c and n-terminal modifications
        cdef double* nmasses = <double *> malloc(self.count_nterm_ions * sizeof(double))
        cdef double* cmasses = <double *> malloc(self.count_cterm_ions * sizeof(double))
        memcpy(nmasses,&self.nterm_ion_masses[0],self.count_nterm_ions * sizeof(double))
        memcpy(cmasses,&self.cterm_ion_masses[0],self.count_cterm_ions * sizeof(double))
        cdef double pepmass
        cdef int i
        for i in range(self.count_nterm_ions):
            nmasses[i] += self.proton_mass
        for i in range(self.count_cterm_ions):
            cmasses[i] += self.proton_mass

        cdef int lastidx = fragment_count - 1

        # translate the residues to masses
        cdef int r
        for r in range(peptide_length):
            residue_mass[r] = self.amino_acids_masses[base_sequence[r]] + \
                              self.modification_masses[modifications[r + 2]]
        # add terminal modification masses
        if modifications[0]:
            residue_mass[0] += self.modification_masses[modifications[0]]
        if modifications[1]:
            residue_mass[peptide_length - 1] += self.modification_masses[modifications[1]]

        # make separate space for one set of c and n terminal fragments - independent of actual
        # fragment type

        cdef int nextrow = 0
        # now iterate over the residues
        cdef int cion
        cdef int nion
        cdef int cres = peptide_length - 1
        if self.single_ion_mode:
            for r in range(peptide_length - 1):
                # next n-terminal mass
                nmasses[0] += residue_mass[r]
                self._fill_fragment_row_minimal(return_frags, nextrow, r + 1, nmasses[0], b'n',
                                        self.nterm_ion_names[0], 0, r + 1)
                nextrow += 1

                cmasses[0] += residue_mass[cres]
                self._fill_fragment_row_minimal(return_frags, nextrow, r + 1, cmasses[0], b'c',
                                        self.cterm_ion_names[0], cres, peptide_length)
                nextrow += 1
                cres -= 1

        else:
            for r in range(peptide_length - 1):
                # next n-terminal mass
                for nion in range(self.count_nterm_ions):
                    nmasses[nion] += residue_mass[r]
                    self._fill_fragment_row_minimal(return_frags, nextrow, r + 1, nmasses[nion], b'n',
                                            self.nterm_ion_names[nion], 0, r + 1)
                    nextrow += 1

                for cion in range(self.count_cterm_ions):
                    cmasses[cion] += residue_mass[cres]
                    self._fill_fragment_row_minimal(return_frags, nextrow, r + 1, cmasses[cion], b'c',
                                            self.cterm_ion_names[cion], cres, peptide_length)
                    nextrow += 1
                cres -= 1

        if include_precursor:
            self._fill_fragment_row_minimal(return_frags, lastidx, peptide_length,
                                    peptide_mass, b'P', b'P', 0, peptide_length)

        free(residue_mass)
        free(nmasses)
        free(cmasses)
        return return_frags_array



    cpdef np.ndarray fragment(self, unsigned char* base_sequence,
                 np.uint8_t[:] modifications,
                 int peptide_length,
                 int include_precursor,
                 double peptide_mass):
        """
        Python callable wrapper for _fragment - mainly for testing.
        
        :param base_sequence - the unmodified peptide sequence
        :param modifications - modification indexes in format
                            nterm,cterm,res0,res1,res2...
                            0 means no modification
                            1 first modification defined in the config
                            2 second modification defined in the config
                            ...
        :param peptide_length - number of residues in the peptide
        :param include_precursor - Whether to include the precursor or not
        :param peptide_mass - singly charged mass of the peptide to be fragmented
                              only used for precursor ion generation
        :return: fragment array
        """
        return self._fragment(base_sequence, modifications, peptide_length,
                                    include_precursor, peptide_mass)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cdef inline void _fill_fragment_row_minimal(self, cfragment_minimal_dtype[:] frags, int  row,
                                                int idx, double single_charge_mz,
                                                unsigned char term, unsigned char iontype,
                                                np.uint8_t r_from, np.uint8_t r_to):
        """
        Fill a single row of the fragment-table with the given values.
        
        :param frags - the fragment-table
        :param row - into which row to store fragment
        :param idx - the index of the fragment (e.g. 9 for y9 ion)
        :param single_charge_mz - the expected single charged m/z value for the fragment 
        :param term - b'n' if it is an n-terminal fragment b'c' if cterminal
        :param iontype - ion type of the fragment, e.g. b, y or P
        :param r_from - start of the fragment range
        :param r_to - end of the fragment range
        """
        frags[row].mz = single_charge_mz
        frags[row].ion_type = iontype
        frags[row].term = term
        frags[row].idx = idx
        frags[row].start = r_from
        frags[row].end = r_to


# defined that the Class defintion should be exported from this file
exports = [Fragmentation]
