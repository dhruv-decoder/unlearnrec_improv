"""Unit tests for config/params.py — str2bool and ParseArgs."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pytest

from config.params import str2bool


class TestStr2Bool:
    @pytest.mark.parametrize("val", [True, False])
    def test_passthrough_bool(self, val):
        assert str2bool(val) is val

    @pytest.mark.parametrize("val", ["yes", "true", "t", "y", "1",
                                      "YES", "True", "T", "Y"])
    def test_truthy_strings(self, val):
        assert str2bool(val) is True

    @pytest.mark.parametrize("val", ["no", "false", "f", "n", "0",
                                      "NO", "False", "F", "N"])
    def test_falsy_strings(self, val):
        assert str2bool(val) is False

    def test_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("maybe")

    def test_invalid_empty(self):
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("")


class TestParseArgs:
    def test_defaults(self):
        # ParseArgs() reads sys.argv; we need to clear it to get defaults
        original_argv = sys.argv
        sys.argv = ["test"]
        try:
            from config.params import ParseArgs
            parsed = ParseArgs()
            assert parsed.lr == pytest.approx(1e-3)
            assert parsed.batch == 512
            assert parsed.epoch == 200
            assert parsed.latdim == 128
            assert parsed.gnn_layer == 3
            assert parsed.topk == 20
            assert parsed.model == "simgcl"
            assert parsed.seed == 1234
        finally:
            sys.argv = original_argv

    def test_custom_args(self):
        original_argv = sys.argv
        sys.argv = ["test", "--lr", "0.01", "--batch", "1024", "--epoch", "50"]
        try:
            from config.params import ParseArgs
            parsed = ParseArgs()
            assert parsed.lr == pytest.approx(0.01)
            assert parsed.batch == 1024
            assert parsed.epoch == 50
        finally:
            sys.argv = original_argv

    def test_str2bool_args(self):
        original_argv = sys.argv
        sys.argv = ["test", "--adversarial_attack", "true", "--allgrad", "false"]
        try:
            from config.params import ParseArgs
            parsed = ParseArgs()
            assert parsed.adversarial_attack is True
            assert parsed.allgrad is False
        finally:
            sys.argv = original_argv
