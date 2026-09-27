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
    # give time to create the file (up to 5 seconds)
    for _ in range(50):
        if os.path.exists(out_file):
            break
        time.sleep(0.1)
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


def _read_log(out_file, expected_line, timeout=5):
    """Wait until expected_line was written to the log file and return all lines."""
    for _ in range(timeout * 10):
        if os.path.exists(out_file):
            with open(out_file) as f:
                lines = f.read().splitlines()
            if any(expected_line in line for line in lines):
                return lines
        time.sleep(0.1)
    raise AssertionError(f"'{expected_line}' not found in log file")


@pytest.mark.parametrize('progress', [False, True])
def test_progress_bar_logs_eta(tmpdir, monkeypatch, progress):
    """
    Progress lines with ETA are written to the log file, with or without progress bars.
    Will be run twice once with progress=False and once with progress=True.
    """
    import xicommon.xi_logging as xi_logging
    out_file = str(tmpdir + "/xi_progress_log.txt")
    clock = [1000.0]
    monkeypatch.setattr(xi_logging, 'time', lambda: clock[0])
    monkeypatch.setattr(xi_logging, '_start_time', 1000.0)
    log_enable(False)
    xi_logging.progress_enable(progress)
    log_file(out_file, threaded=True)
    try:
        bar = xi_logging.ProgressBar("Digesting things", 200)
        # 1 item per second: a line every 11 seconds, the last one after 99 items
        for _ in range(100):
            clock[0] += 1
            bar.next()
        # no new percent, but more than log_max_interval seconds later
        clock[0] += xi_logging.ProgressBar.log_max_interval + 1
        bar.next(0)
        clock[0] += 5
        bar.finish()
        lines = _read_log(out_file, "Digesting things finished")
    finally:
        log_file(False)
        xi_logging.progress_enable(False)

    assert sum(line.endswith(": Digesting things") for line in lines) == 1
    assert "99.000: Digesting things 49% ~101s remaining (99/200)" in lines
    # heartbeat line without a new percent, ETA from the now slower average rate
    assert "701.000: Digesting things 50% ~11m remaining (100/200)" in lines
    assert "706.000: Digesting things finished (0:11:46 total)" in lines
    # at most one progress line per log_interval seconds
    progress_lines = [line for line in lines if "remaining" in line]
    times = [float(line.split(':')[0]) for line in progress_lines]
    assert all(b - a > xi_logging.ProgressBar.log_interval for a, b in zip(times, times[1:]))


def test_format_eta():
    from xicommon.xi_logging import format_eta
    assert format_eta(42) == "42s"
    assert format_eta(121) == "2m"
    assert format_eta(7201) == "2h"
    assert format_eta(172801) == "2days"
