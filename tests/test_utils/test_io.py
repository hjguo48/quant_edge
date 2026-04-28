from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.io import write_json_atomic


def test_write_json_atomic_writes_complete_payload(tmp_path: Path) -> None:
    target = tmp_path / "state.json"

    write_json_atomic(target, {"status": "ok", "value": 7})

    assert json.loads(target.read_text()) == {"status": "ok", "value": 7}


def test_write_json_atomic_preserves_existing_file_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    target.write_text(json.dumps({"status": "old", "value": 1}))

    def interrupted_dump(payload, handle, *, indent, sort_keys):  # type: ignore[no-untyped-def]
        handle.write('{"status": ')
        handle.flush()
        raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr("src.utils.io.json.dump", interrupted_dump)

    with pytest.raises(KeyboardInterrupt):
        write_json_atomic(target, {"status": "new", "value": 2})

    assert json.loads(target.read_text()) == {"status": "old", "value": 1}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []
