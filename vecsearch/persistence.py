"""Saving and loading indexes.

An index is saved as a folder with three files:

    header.json     class name, format version, constructor params (readable)
    arrays.npz      numpy arrays: vectors, ids, LSH planes, ...
    structure.pkl   plain python structures: HNSW layers, LSH buckets, rng state

Big numeric data goes through numpy (compact, fast), and only the python
dicts/lists go through pickle.

Warning: pickle can execute code when loading, so only load index files
you created yourself.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import BaseIndex

FORMAT_VERSION = 1
HEADER = "header.json"
ARRAYS = "arrays.npz"
STRUCTURE = "structure.pkl"


def _registry() -> dict[str, type["BaseIndex"]]:
    """Map class names to classes (imported here to avoid circular imports). O(1)."""
    from .brute_force import BruteForceIndex
    from .hnsw import HNSWIndex
    from .lsh import LSHIndex

    return {cls.__name__: cls for cls in (BruteForceIndex, LSHIndex, HNSWIndex)}


def save_index(index: "BaseIndex", path: str | Path) -> Path:
    """Write ``index`` into the folder ``path`` (created if needed).

    Time: O(n * d + size of graph/buckets)   Space: O(1) extra besides the file buffers.

    Returns:
        the folder path.
    """
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    state = index._get_state()

    header = {
        "format_version": FORMAT_VERSION,
        "class": type(index).__name__,
        "size": len(index),
        "params": state["params"],
    }
    (folder / HEADER).write_text(json.dumps(header, indent=2), encoding="utf-8")
    np.savez(folder / ARRAYS, **state["arrays"])
    with open(folder / STRUCTURE, "wb") as f:
        pickle.dump(state["extra"], f, protocol=pickle.HIGHEST_PROTOCOL)
    return folder


def load_index(path: str | Path) -> "BaseIndex":
    """Rebuild an index from a folder written by ``save_index``.

    Time: O(n * d + size of graph/buckets)   Space: same as the index.

    Raises:
        FileNotFoundError: folder or one of the files is missing.
        ValueError: unknown class or unsupported format version.
    """
    folder = Path(path)
    for name in (HEADER, ARRAYS, STRUCTURE):
        if not (folder / name).exists():
            raise FileNotFoundError(f"{folder / name} not found")

    header = json.loads((folder / HEADER).read_text(encoding="utf-8"))
    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported index format version {header.get('format_version')}")
    classes = _registry()
    cls_name = header.get("class")
    if cls_name not in classes:
        raise ValueError(f"unknown index class {cls_name!r}")

    with np.load(folder / ARRAYS, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    with open(folder / STRUCTURE, "rb") as f:
        extra = pickle.load(f)  # noqa: S301 - only load your own files

    index = classes[cls_name](**header["params"])
    index._set_state({"params": header["params"], "arrays": arrays, "extra": extra})
    if len(index) != header["size"]:
        raise ValueError("index size does not match header, files may be corrupted")
    return index
