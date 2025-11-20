#cython: language_level=3

cimport numpy as np

cdef class byte_ranges_context:
    cdef const np.uint8_t[:, :] src
    cdef const np.intp_t[:] src_indices
    cdef const np.intp_t[:] starts
    cdef const np.intp_t[:] ends

cdef int compare_byte_ranges(const void *a, const void *b, void *arg) noexcept
