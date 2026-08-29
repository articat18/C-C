"""Atomic checkpoint storage for deterministic experiment runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from experiment_boundary import REPOSITORY_ROOT, resolve_editable_path


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely stored or loaded."""


class CheckpointManager:
    """Store model members inside each experiment package."""

    def __init__(self, root: str | Path = "experiments") -> None:
        self.root = resolve_editable_path(root)

    def experiment_dir(self, experiment_id: str) -> Path:
        return resolve_editable_path(self.root / experiment_id / "checkpoints")

    def save_member(
        self,
        experiment_id: str,
        member: int,
        *,
        model: Any,
        metadata: dict[str, Any],
    ) -> dict[str, str]:
        directory = self.experiment_dir(experiment_id)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"member-{member:02d}"
        arrays_path = directory / f"{stem}.npz"
        metadata_path = directory / f"{stem}.json"
        if arrays_path.exists() or metadata_path.exists():
            raise CheckpointError(f"checkpoint already exists: {arrays_path}")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{stem}-", suffix=".npz", dir=directory
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(
                    stream,
                    V=model.V,
                    W=model.W,
                    b=np.asarray(model.b, dtype=np.float32),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, arrays_path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        _atomic_json_write(metadata_path, metadata)
        return {
            "arrays": arrays_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "metadata": metadata_path.relative_to(REPOSITORY_ROOT).as_posix(),
        }

    def load_member(self, experiment_id: str, member: int) -> dict[str, Any]:
        directory = self.experiment_dir(experiment_id)
        stem = f"member-{member:02d}"
        arrays_path = directory / f"{stem}.npz"
        metadata_path = directory / f"{stem}.json"
        if not arrays_path.is_file() or not metadata_path.is_file():
            raise CheckpointError(f"checkpoint is incomplete: {directory / stem}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            state = {name: arrays[name].copy() for name in ("V", "W", "b")}
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        return {"state": state, "metadata": metadata}


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
