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

"""Module that handles output (log messages and progress bars)."""
from time import time
from datetime import timedelta
from progress.bar import Bar
from multiprocessing import Queue, Process
import queue as Q
import threading
import os
from pathlib import Path


_log_enabled = False
_log_file = False
_progress_enabled = False
_start_time = time()
# queue used to forward log entries to the file-writer
_log_queue = None
# the process that is doing the writing
_log_file_process = None


def log_timestamp_reset():
    """Reset the log time to the current time."""
    global _start_time
    _start_time = time()


def log_enable(setting):
    """Enable or disable logging."""
    global _log_enabled
    _log_enabled = bool(setting)


def log_queue_writer(queue, file):
    """Function that will run as process receiving the log and writing it to a file."""
    # create the parent folder of the log-file
    parent_folder = os.path.dirname(file)
    if parent_folder:
        Path(parent_folder).mkdir(parents=True, exist_ok=True)
    # open the output file
    with open(file, "a", encoding="utf-8") as log_out:
        # start reading from queue
        while True:
            s = queue.get()
            if s:
                log_out.write(s)
                log_out.write("\n")
                log_out.flush()
            else:
                # take a False as an indication to close down the process
                log_out.write("Stopping log writer process\n")
                log_out.flush()
                if hasattr(queue, "close"):
                    queue.close()
                if hasattr(queue, "join_thread"):
                    queue.join_thread()
                break


def log_file(file, threaded=False):
    """
    Define that the log should be writen out to a file.

    :param file - (str,False) if a string then it defines the output path; if False disables writing
    :param threaded - (bool) if True uses a thread instead of a process for writing
    """
    global _log_file
    global _log_queue
    global _log_file_process

    if isinstance(file, str):
        _log_file = True
        if _log_file_process is not None:
            # close down the old process
            _log_queue.put(False)
            _log_file_process.join(30)
            _log_file_process = None
        if threaded:
            _log_queue = Q.Queue()
        else:
            _log_queue = Queue()

        # start the log writer process
        # log_queue_writer MUST be at module level to be picklable on macOS/Windows
        if threaded:
            _log_file_process = threading.Thread(
                target=log_queue_writer,
                args=(_log_queue, file),
                daemon=True
            )
        else:
            _log_file_process = Process(
                target=log_queue_writer,
                args=(_log_queue, file),
                daemon=True
            )
        _log_file_process.start()
    elif isinstance(file, bool) and not file:
        _log_file = False
        if _log_queue is not None:
            # close down the old process
            _log_queue.put(False)
            if _log_file_process is not None:
                _log_file_process.join()
                _log_file_process = None
        _log_queue = None
    else:
        raise ValueError("log_file only accepts a file path or False as parameter")


def progress_enable(setting):
    """Enable or disable displaying progress bars."""
    global _progress_enabled
    _progress_enabled = bool(setting)


def log(message):
    """Log a message."""
    if _log_enabled or _log_file:
        timestamp = time() - _start_time
        timedmessage = "%.3f: %s" % (timestamp, message)
        if _log_enabled:
            if message:
                print(timedmessage)
            else:
                print(flush=True)
        if _log_file:
            if message:
                _log_queue.put(timedmessage)
            else:
                log_file(False)


def format_eta(eta):
    """
    Format a remaining time in a human readable way.

    :param eta: (float) remaining time in seconds
    :return: (str) e.g. '3days', '5h', '12m' or '42s'
    """
    eta = int(eta)
    # more then two days left
    if eta > 172800:
        return str(eta // 86400) + "days"
    # more than 2 hours - report in hours
    if eta > 7200:
        return str(eta // 3600) + "h"
    # more than two minutes - report in minutes
    if eta > 120:
        return str(eta // 60) + "m"
    # otherwise report in seconds
    return str(eta) + "s"


class ProgressBar(object):
    """Bar to visualize the progression of a process."""

    # minimum time in seconds between progress lines in the log file
    log_interval = 10
    # time in seconds after which a progress line is logged even without a new percent
    log_max_interval = 600

    class NiceEtaBar(Bar):
        len_last_eta = 0

        def __init__(self, *args, **kwargs):
            """
            Forward everything to Bar.__init().
            """
            super().__init__(*args, **kwargs)

        @property
        def nice_eta(self):
            """
            Transform the eta to a nicer human readable format.
            """
            if self.index == self.max:
                ret = str(self.elapsed_td) + " total"
            else:
                ret = str(int(self.percent*10)/10) + "% ~" + format_eta(self.eta) + \
                    "  remaining (" + str(self.index) + "/" + str(self.max) + ")"

            # clean up left over from last print out
            new_len = len(ret)
            if new_len > self.len_last_eta:
                ret += " " * (new_len - self.len_last_eta)

            self.len_last_eta = new_len
            return ret

    def __init__(self, message, total):
        """Initialise the ProgressBar with a message and a total number."""
        self.message = message
        self.total = total
        self.count = 0
        self.percent = 0
        self.log_percent = 0
        self.start = time() - _start_time
        self.timestamp = self.start
        self.logtimestamp = self.start
        self.bar = None
        if _progress_enabled:
            if total > 1:
                self.bar = self.NiceEtaBar("%.3f: %s" % (self.timestamp, message), max=total,
                                           suffix='%(nice_eta)s')
            if _log_file:
                _log_queue.put("%.3f: %s" % (self.timestamp, message))
        else:
            # writes to the log file as well
            log(message)

    def next(self, add_to_count=1):
        """Progress the bar."""
        if self.total <= 1 or (self.bar is None and not _log_file):
            return
        self.count += add_to_count
        # current timestamp
        timestamp = time() - _start_time
        percent = int(self.count / self.total * 100)
        # update the progress bar if we passed a new percent
        # but at most ones per second
        # but also if more then a minute has passed
        if self.bar is not None and (((percent > self.percent) and (timestamp - self.timestamp > 1))
                                     or (timestamp - self.timestamp > 60)):
            self.timestamp = timestamp
            self.bar.goto(self.count)
            self.percent = percent
        # write progress to the log file - independent of the progress bar being shown.
        # Not faster then every log_interval seconds and only for a new percent (at most 100
        # times), unless log_max_interval seconds have passed
        if _log_file and self.count > 0:
            since_log = timestamp - self.logtimestamp
            if (since_log > self.log_interval and percent > self.log_percent) \
                    or since_log > self.log_max_interval:
                self.logtimestamp = timestamp
                self.log_percent = percent
                # average rate since the start - steadier than the bar's moving average
                eta = (timestamp - self.start) / self.count * (self.total - self.count)
                _log_queue.put("%.3f: %s %i%% ~%s remaining (%i/%i)" % (
                    timestamp, self.message, percent, format_eta(eta), self.count, self.total))

    def finish(self):
        """Finish the ProgressBar."""
        if self.bar is not None:
            self.bar.goto(self.total)
            self.bar.finish()
        if _log_file:
            timestamp = time() - _start_time
            _log_queue.put("%.3f: %s finished (%s total)" % (
                timestamp, self.message, timedelta(seconds=int(timestamp - self.start))))
