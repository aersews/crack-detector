from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class CheckpointStore:
    """Crash-safe pipeline state persisted with atomic replace + fsync."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": self.SCHEMA_VERSION,
                "created_at": time.time(),
                "updated_at": time.time(),
                "status": "not_started",
                "completed": {},
                "failed": {},
            }
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("Unsupported checkpoint schema version")
            return state
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            raise RuntimeError(f"Checkpoint file is unreadable: {self.path}") from exc

    def _atomic_write(self) -> None:
        payload = json.dumps(self.state, indent=2, sort_keys=True, default=str).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            try:
                dir_fd = os.open(self.path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (AttributeError, OSError):
                pass
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def save(self) -> None:
        with self._lock:
            self.state["updated_at"] = time.time()
            self._atomic_write()

    def set_status(self, status: str) -> None:
        self.state["status"] = status
        self.save()

    def is_completed(self, key: str) -> bool:
        return key in self.state["completed"]

    def get_completed(self, key: str) -> dict[str, Any] | None:
        return self.state["completed"].get(key)

    def mark_completed(self, key: str, **metadata: Any) -> None:
        with self._lock:
            self.state["completed"][key] = {"completed_at": time.time(), **metadata}
            self.state["failed"].pop(key, None)
            self.save()

    def mark_failed(self, key: str, error: str) -> None:
        with self._lock:
            self.state["failed"][key] = {"failed_at": time.time(), "error": error}
            self.save()

    def completed_keys(self) -> set[str]:
        return set(self.state["completed"])

    def reset(self) -> None:
        with self._lock:
            self.state = {
                "schema_version": self.SCHEMA_VERSION,
                "created_at": time.time(),
                "updated_at": time.time(),
                "status": "not_started",
                "completed": {},
                "failed": {},
            }
            self.save()
