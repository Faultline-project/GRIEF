"""Versioned, append-only run events shared by the CLI, TUI and playback."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    return str(value)


class EventWriter:
    def __init__(self, path: str = ""):
        self.path = Path(path) if path else None
        self.run_id = uuid.uuid4().hex[:12]
        self.seq = 0
        self.started = time.monotonic()

    def emit(self, kind: str, **data):
        if self.path is None:
            return
        self.seq += 1
        record = {"schema_version": 1, "run_id": self.run_id, "seq": self.seq,
                  "type": kind, "elapsed_s": round(time.monotonic() - self.started, 3),
                  "data": data}
        # One writer per run. Close after each line so readers can tail immediately.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, default=json_default) + "\n")


class EventReader:
    """Incremental reader; retain incomplete lines until the writer finishes them."""
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.errors = 0

    def read(self, limit=200):
        if not self.path.exists():
            return []
        events = []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            for _ in range(limit):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    break
                self.offset = stream.tell()
                try:
                    item = json.loads(line)
                    if item.get("schema_version") == 1 and isinstance(item.get("data"), dict):
                        events.append(item)
                    else:
                        self.errors += 1
                except (ValueError, AttributeError):
                    self.errors += 1
        return events
