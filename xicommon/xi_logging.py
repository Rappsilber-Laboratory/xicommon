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
from progress.bar import Bar
from multiprocessing import Queue, Process
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


def log_file(file):
    """
    Define that the log should be writen out to a file.

    :param file - (str,False) if a string then it defines the output path; if False disables writing
    """
    global _log_file
    global _log_queue
    global _log_file_process

    if isinstance(file, str):
        _log_file = True
        if _log_queue is not None:
            # close down the old process
            _log_queue.put(False)
        _log_queue = Queue()

        # function that will run as process receiving the log and writing it to a file
        def log_queue_writer(queue):
            # create the parent folder of the log-file
            parent_folder = os.path.dirname(file)
            Path(parent_folder).mkdir(parents=True, exist_ok=True)
            # open the output file
            log_out = open(file, "a")

            # start reading from queue
            while True:
                s = queue.get()
                if s:
                    log_out.write(s)
                    log_out.write("\n")
                    log_out.flush()
                else:
                    # take a False as an indication to close down the process
                    break
            log_out.close()

        # start the log writer process
        # TODO: fails on windows due to second level local function (log_queue_writer) cannot be
        # pickled
        _log_file_process = Process(target=log_queue_writer, args=(_log_queue,), daemon=True)
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


class ProgressBar(object):
    """Bar to visualize the progression of a process."""
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
                ret = str(int(self.percent*10)/10) + "% ~"
                eta = self.eta
                # more then two days left
                if eta > 172800:
                    ret += str(eta // 86400) + "days  remaining (" + str(self.index) + "/" + str(
                        self.max) + ")"
                # more than 2 hours - report in hours
                elif eta > 7200:
                    ret += str(eta // 3600) + "h  remaining (" + str(self.index) + "/" + str(
                        self.max) + ")"
                # more than two minutes - report in minutes
                elif eta > 120:
                    ret += str(eta // 60) + "m  remaining (" + str(self.index) + "/" + str(
                        self.max) + ")"
                # otherwise report in seconds
                else:
                    ret += str(eta) + "s  remaining (" + str(self.index) + "/" + str(self.max) + ")"

            # clean up left over from last print out
            new_len = len(ret)
            if new_len > self.len_last_eta:
                ret += " " * (new_len - self.len_last_eta)

            self.len_last_eta = new_len
            return ret

    def __init__(self, message, total):
        """Initialise the ProgressBar with a message and a total number."""
        self.message = message
        if _progress_enabled:
            self.timestamp = time() - _start_time
            self.logtimestamp = self.timestamp
            if total > 1:
                self.bar = self.NiceEtaBar("%.3f: %s" % (self.timestamp, message), max=total,
                                           suffix='%(nice_eta)s')
            self.total = total
            self.count = 0
            self.percent = 0
            if _log_file:
                _log_queue.put("%.3f: %s" % (self.timestamp, message))
        else:
            log(message)

    def next(self, add_to_count=1):
        """Progress the bar."""
        if _progress_enabled and self.total > 1:
            self.count += add_to_count
            # current timestamp
            timestamp = time() - _start_time
            percent = int(self.count / self.total * 100)
            # update the progress bar if we passed a new percent
            # but at most ones per second
            # but also if more then a minute has passed
            if ((percent > self.percent) and (timestamp - self.timestamp > 1)) \
                    or (timestamp - self.timestamp > 60):
                self.timestamp = timestamp
                self.bar.goto(self.count)
                if _log_file:
                    # don't write the progress-update faster then ones every ten seconds and at
                    # most 100 times
                    if (timestamp - self.logtimestamp > 10) and (percent > self.percent):
                        self.logtimestamp = timestamp
                        _log_queue.put("%.3f: %s %i%%" % (self.timestamp, self.message, percent))
                self.percent = percent

    def finish(self):
        """Finish the ProgressBar."""
        if _progress_enabled and self.total > 1:
            self.bar.goto(self.total)
            self.bar.finish()
        if _log_file:
            timestamp = time() - _start_time
            _log_queue.put("%.3f: %s finished" % (timestamp, self.message))
