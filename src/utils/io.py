from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_json_atomic(
    path: Path | str,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """Write JSON to ``path`` atomically."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        temp_path = None
        try:
            os.chmod(output_path, 0o644)
        except OSError as exc:
            warnings.warn(
                f"write_json_atomic could not chmod {output_path} to 0o644: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass
