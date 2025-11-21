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

"""Module to centralise import of all cython routines."""
from importlib import import_module
import numpy as np
import pyximport
import os

# Try to avoid the NPY deprecation warnings at compile time via CPPFLAGS
_npy_define = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'
# don't clobber  existing CPPFLAGS environment variable.  
# Only prepend the define if not present.
_existing_cppflags = os.environ.get('CPPFLAGS', '')
if _npy_define not in _existing_cppflags:
    os.environ['CPPFLAGS'] = (_npy_define + ' ' + _existing_cppflags).strip() 

directory = os.path.dirname(__file__)
pyximport.install(setup_args={'include_dirs': np.get_include()}, inplace=True)
files = os.listdir(directory)

__all__ = []


def _get_name_for_attribute(search_module, search_attribute):
    """
    Return the name of an attribute.

    This is used to be able to have an export variable in each cython file that define what should
    be exported.

    :param search_module - the module that contains the attribute
    :param search_attribute - the attribute of which the name should be found
    """
    for aname in dir(search_module):
        if getattr(search_module, aname) is search_attribute:
            return aname
    return None

restart_needed = False
for filename in files:
    if filename.endswith(".pyx"):
        funcname = filename[:-4]
        modulename = "xicommon.cython." + funcname
        module = import_module(modulename)
        mod_exports = []

        # try to find register array
        try:
            mod_exports = getattr(module, 'exports')
        except AttributeError:
            pass

        # look for the compiled  library and check if it is older than the pyx file?
        file_date = os.path.getmtime(os.path.join(directory, filename))
        
        for cname in files:

            if cname.startswith(funcname) and 'cpython' in cname and \
                not cname.startswith("__") and cname.endswith(".so"):
                so_date = os.path.getmtime(os.path.join(directory, cname))
                # found the so - check date differ for more then 2 seconds
                if so_date < file_date - 2:
                    # so is older - delete it and ask for restart
                    os.remove(os.path.join(directory, cname))
                    print(f"Cython module {modulename} was out of date. Deleted please restart.")
                    restart_needed = True

        # if export was defined
        if len(mod_exports) > 0:
            attrib_to_name = {}
            # go through the exports
            for e in mod_exports:
                # find the name of it
                funcname = _get_name_for_attribute(module, e)
                # and add them to the list of public functions
                globals()[funcname] = e
                __all__.append(funcname)
        else:
            # no "export" list defined - so take the name of the file and export a function with
            # that name
            try:
                globals()[funcname] = getattr(module, funcname)
                __all__.append(funcname)
            except AttributeError:
                pass

if restart_needed:
    raise ImportError(f"Cython modules were out of date and got deleted. Please restart.")

