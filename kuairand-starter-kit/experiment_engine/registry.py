"""Append-only JSON Lines experiment registry."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator

from experiment_boundary import resolve_editable_path

try:  # Unix locking is available in the supported local environments.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback remains single-process safe.
    fcntl = None


class RegistryError(RuntimeError):
    """Raised when registry integrity or uniqueness checks fail."""


class ExperimentRegistry:
    def __init__(self, path: str | Path = "experiments/index.jsonl") -> None:
        self.path = resolve_editable_path(path)

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RegistryError(
                        f"invalid registry JSON at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise RegistryError(
                        f"registry line {line_number} is not a JSON object"
                    )
                yield value

    def contains(self, experiment_id: str) -> bool:
        return any(
            record.get("experiment_id") == experiment_id
            for record in self.records()
        )

    def append(self, record: dict[str, Any]) -> None:
        experiment_id = record.get("experiment_id")
        if not isinstance(experiment_id, str):
            raise RegistryError("registry record requires experiment_id")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a+", encoding="utf-8") as stream:
            with _exclusive_lock(stream):
                stream.seek(0)
                for line in stream:
                    if not line.strip():
                        continue
                    existing = json.loads(line)
                    if existing.get("experiment_id") == experiment_id:
                        raise RegistryError(
                            f"experiment_id is already registered: {experiment_id}"
                        )
                stream.seek(0, os.SEEK_END)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

    def successful_records(self) -> list[dict[str, Any]]:
        return [record for record in self.records() if record.get("status") == "success"]


@contextmanager
def _exclusive_lock(stream: Any) -> Iterator[None]:
    if fcntl is not None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
