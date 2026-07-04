"""Patch sys.argv before any test module imports config.params (which calls
ParseArgs() at module level and would fail on pytest's CLI flags).

Also patches torch.Tensor.cuda to be a no-op so tests can run on CPU-only."""

import sys

# Save original argv and replace with a minimal one so argparse doesn't choke
# on pytest flags like -v, --tb, etc.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]

import torch as t

_original_cuda = t.Tensor.cuda

def _noop_cuda(self, *args, **kwargs):
    return self

t.Tensor.cuda = _noop_cuda
