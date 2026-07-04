"""Unit tests for Utils/time_logger.py."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datetime
import Utils.time_logger as tl


class TestLog:
    def test_log_prints(self, capsys):
        tl.log("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_log_save_true(self):
        tl.logmsg = ""
        tl.log("saved message", save=True)
        assert "saved message" in tl.logmsg

    def test_log_save_false(self):
        tl.logmsg = ""
        tl.log("not saved", save=False)
        assert "not saved" not in tl.logmsg

    def test_log_save_default_true(self, capsys):
        tl.logmsg = ""
        tl.saveDefault = True
        tl.log("default saved")
        assert "default saved" in tl.logmsg
        tl.saveDefault = False  # reset

    def test_log_save_default_false(self):
        tl.logmsg = ""
        tl.saveDefault = False
        tl.log("not default saved")
        assert "not default saved" not in tl.logmsg

    def test_log_oneline(self, capsys):
        tl.log("oneline msg", oneline=True)
        captured = capsys.readouterr()
        assert "oneline msg" in captured.out


class TestMarktime:
    def test_stores_marker(self):
        tl.marktime("test_marker")
        assert "test_marker" in tl.timemark
        assert isinstance(tl.timemark["test_marker"], datetime.datetime)

    def test_overwrites_marker(self):
        tl.marktime("dup")
        first = tl.timemark["dup"]
        tl.marktime("dup")
        second = tl.timemark["dup"]
        assert second >= first
