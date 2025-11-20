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

from xicommon.filters.base_filter import BaseFilter
import pytest


class BadFilter(BaseFilter):
    pass


class GoodFilter(BaseFilter):
    def process(self, spectrum):
        pass


def test_base_filter_is_abc():
    with pytest.raises(TypeError):
        BaseFilter('x')


def test_child_class_fails_when_no_process_method_is_defined():
    with pytest.raises(TypeError):
        BadFilter('x')


def test_base_filter_has_abstract_method_process():
    assert GoodFilter('x')
