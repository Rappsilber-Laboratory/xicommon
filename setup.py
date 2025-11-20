# Copyright (C) 2025  Lutz Fischer
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

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import glob
import os


# Find all .pyx files in the xicommon/cython directory
cython_dir = os.path.join("xicommon", "cython")
pyx_files = glob.glob(os.path.join(cython_dir, "*.pyx"))

# Create Extension objects for each .pyx file
extensions = []
for pyx_file in pyx_files:
    # Get module name (e.g., xicommon.cython.fast_unique)
    module_name = pyx_file.replace(os.sep, ".").replace(".pyx", "")
    
    # Check if file needs C++ compilation
    language = "c++" if "isin_set" in pyx_file else "c"
    extra_compile_args = ["-std=c++11"] if language == "c++" else []
    
    ext = Extension(
        module_name,
        [pyx_file],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language=language,
        extra_compile_args=extra_compile_args,
    )
    extensions.append(ext)

# Cythonize the extensions
ext_modules = cythonize(
    extensions,
    compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'embedsignature': True,
    },
    annotate=False,  # Set to True to generate HTML annotation files for optimization
)

setup(
    ext_modules=ext_modules,
)
