import os
import pickle
import torch as t

_ALLOWED_ROOTS = None


def _get_allowed_roots():
    global _ALLOWED_ROOTS
    if _ALLOWED_ROOTS is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _ALLOWED_ROOTS = [os.path.realpath(project_root)]
    return _ALLOWED_ROOTS


def validate_path(path, allowed_roots=None):
    """Resolve *path* and verify it falls under an allowed directory.

    Prevents path-traversal attacks when file paths originate from CLI
    arguments or other user-controlled input.  Raises ``ValueError`` if
    the resolved path escapes the allowed roots.
    """
    if allowed_roots is None:
        allowed_roots = _get_allowed_roots()
    real = os.path.realpath(path)
    for root in allowed_roots:
        if real.startswith(root + os.sep) or real == root:
            return real
    raise ValueError(
        f"Path '{path}' resolves to '{real}' which is outside allowed directories: {allowed_roots}"
    )


def safe_pickle_load(path):
    """Load a pickle file after validating the path is inside the project."""
    path = validate_path(path)
    with open(path, 'rb') as fs:
        return pickle.load(fs)


def safe_torch_load(path):
    """Load a torch checkpoint with ``weights_only=True`` after path validation.

    ``weights_only=True`` prevents arbitrary code execution via crafted
    checkpoint files.  If the checkpoint was saved with ``pickle``-based
    objects (e.g. full ``nn.Module`` instances), torch will raise an error;
    migrate to saving/loading ``state_dict`` instead for full safety.

    For backwards compatibility with existing checkpoints that embed full
    model objects, this falls back to ``weights_only=False`` when the
    safe load fails with an ``UnpicklingError``.  A warning is printed
    so callers can migrate their saved checkpoints.
    """
    path = validate_path(path)
    try:
        return t.load(path, weights_only=True)
    except Exception:
        import warnings
        warnings.warn(
            f"Could not load '{path}' with weights_only=True. "
            "Falling back to weights_only=False. "
            "Re-save checkpoints using state_dict for full safety.",
            stacklevel=2,
        )
        return t.load(path, weights_only=False)


def safe_torch_save(content, path):
    """Save a torch checkpoint after validating the destination path."""
    path = validate_path(path)
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    t.save(content, path)
