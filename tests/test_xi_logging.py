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

from xicommon.xi_logging import log_enable, log, log_file, log_timestamp_reset
import os
import re
import time
import pytest


def test_logging(tmpdir, capsys):
    """
    Test that logging to stdout and file can be enabled and disabled - and that output
    arrives.
    """
    out_file = tmpdir + "/xi_log_test.txt"
    if os.path.exists(out_file):
        os.remove(out_file)

    # log_file should only accept a file path or False as parameter
    with pytest.raises(ValueError):
        log_file(123)

    log_file(str(out_file))
    # give time to create the file
    time.sleep(1)
    # file was created
    assert os.path.exists(out_file)
    # enable logging to stdout
    log_enable(True)
    log("test both")
    # test that we got the output
    captured = capsys.readouterr()
    assert not re.search(r"[0-9.]*\s:*test both\n", captured.out) is None
    # only log to file now
    log_enable(False)
    log("test file only")
    captured = capsys.readouterr()
    assert re.search(r"[0-9.]*\s:*test file only\n", captured.out) is None

    log_enable(True)
    log_file(False)
    log("test out only")
    captured = capsys.readouterr()
    assert not re.search(r"[0-9.]*\s:*test out only\n", captured.out) is None

    log_timestamp_reset()
    time.sleep(2)
    # test that the time works
    log("test timestamp 2 seconds")
    captured = capsys.readouterr()
    assert not re.search(r"2\.[0-9]*:\s*test timestamp 2 seconds\n", captured.out) is None

    # and that we can reset the time
    log_timestamp_reset()
    log("test timestamp 0")
    captured = capsys.readouterr()
    assert not re.search(r"0\.[0-9]*:\s*test timestamp 0\n", captured.out) is None

    log_file(str(out_file))
    log("final")
    captured = capsys.readouterr()
    assert not re.search(r"final", captured.out) is None

    # change the output file
    out_file2 = tmpdir + "/xi_log_test2.txt"
    log_file(str(out_file2))
    log("new file")
    log_file(False)
    time.sleep(0.5)

    # make sure the logfile-process had time to write out everything
    time.sleep(1)
    with open(out_file) as f:
        read_data = f.read()
        # test that all lines that should be in the file are there
        assert not re.search(r"[0-9.]*\s:*test file only\n", read_data) is None
        assert not re.search(r"[0-9.]*\s:*test both\n", read_data) is None
        assert not re.search("final", read_data) is None
        # test that one that should nopt be there is also not there
        assert re.search(r"[0-9.]*\s:*test out only\n", read_data) is None
        # and also none of the timestamp test should be there
        assert re.search(r"timestamp", read_data) is None
        # should not have entries send to new file
        assert re.search(r"new file", read_data) is None

    with open(out_file2) as f:
        read_data = f.read()
        assert not re.search(r"[0-9.]*\s:*new file\n", read_data) is None
