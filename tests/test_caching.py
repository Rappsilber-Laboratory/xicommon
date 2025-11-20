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

from xicommon.cache import Cache
import os


def test_cache(tmpdir):

    cache = Cache(tmpdir)

    item = cache.item('test', 0, 0xDEADBEEF, '.ext')

    assert item.name == 'test'
    assert item.hash == 'deadbeef'
    assert item.load_filename == os.path.join(tmpdir, 'test-deadbeef-v0.ext')
    assert item.save_filename == os.path.join(tmpdir, 'test-deadbeef-v0.partial.ext')
    assert not item.exists

    open(item.save_filename, 'w').close()

    assert not item.exists

    item.validate()

    assert item.exists

    assert os.path.exists(item.load_filename)
