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

"""
xicommon: Common library for XiSearch mass spectrometry tools.

This package contains shared functionality used by both xisearch-search and xisearch-annotator:
- Configuration system (config, const, dtypes)
- Fragmentation (fragmentation, fragment_peptides, modifications, mass)
- Filters (filters/)
- Cython optimizations (cython/)
- I/O utilities (spectra_reader, cache, utils, xi_logging, output_format)
- Context base classes (context_base, simple_databases, mock_context)
"""

__version__ = "1.0.0"

# Core modules
from . import config
from . import const
from . import dtypes
from . import fragmentation
from . import fragment_peptides
from . import modifications
from . import mass
from . import spectra_reader
from . import cache
from . import utils
from . import xi_logging
from . import output_format
from . import context_base
from . import simple_databases
from . import mock_context
from . import synthetic_spectra

# Subpackages
from . import filters
from . import cython

__all__ = [
    "config",
    "const",
    "dtypes",
    "fragmentation",
    "fragment_peptides",
    "modifications",
    "mass",
    "spectra_reader",
    "cache",
    "utils",
    "xi_logging",
    "output_format",
    "context_base",
    "simple_databases",
    "mock_context",
    "synthetic_spectra",
    "filters",
    "cython",
]
