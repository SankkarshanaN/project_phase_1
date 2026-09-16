"""Eager .npz loading.

`np.load` on an .npz returns a lazy `NpzFile` that keeps the underlying zip
handle open until it is closed or garbage-collected. That is fine for a handful
of frames and fatal for a dataset: holding one per frame across 15,420 frames
exhausts the process file-handle limit and the run dies partway through with
`OSError [Errno 24] Too many open files`. It happened in three separate scripts
because the lazy handle is invisible at the call site -- the object looks like a
dict and is used like one.

`load_npz` reads the arrays out and closes the file, so callers can keep as many
frames in memory as they have memory for.
"""
from pathlib import Path

import numpy as np


def load_npz(path, keys=None, allow_pickle: bool = True) -> dict:
    """Reads an .npz into a plain dict of arrays, closing the file.

    `keys` restricts what is read, which matters for the recorded frames: the
    BEV grids and radar returns dominate their size, and a check that only
    needs the per-object arrays should not pay for them. Missing keys are
    skipped rather than raising, since older frames carry fewer fields.
    """
    with np.load(Path(path), allow_pickle=allow_pickle) as z:
        wanted = z.files if keys is None else [k for k in keys if k in z.files]
        return {k: z[k] for k in wanted}
