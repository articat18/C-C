from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from experiment_engine.checkpoints import CheckpointError, CheckpointManager
from experiment_engine.registry import ExperimentRegistry, RegistryError
from experiment_boundary import REPOSITORY_ROOT


class ExperimentStorageTests(unittest.TestCase):
    def setUp(self):
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="storage-test-", dir=runs)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_checkpoint_round_trip_and_no_overwrite(self):
        manager = CheckpointManager(self.root / "checkpoints")
        model = SimpleNamespace(
            V=np.arange(12, dtype=np.float32).reshape(6, 2),
            W=np.arange(6, dtype=np.float32),
            b=np.float32(0.25),
        )
        manager.save_member("E0001", 0, model=model, metadata={"seed": 3})
        loaded = manager.load_member("E0001", 0)
        np.testing.assert_array_equal(loaded["state"]["V"], model.V)
        np.testing.assert_array_equal(loaded["state"]["W"], model.W)
        self.assertAlmostEqual(float(loaded["state"]["b"]), float(model.b))
        self.assertEqual(loaded["metadata"], {"seed": 3})
        with self.assertRaises(CheckpointError):
            manager.save_member("E0001", 0, model=model, metadata={"seed": 3})

    def test_registry_is_append_only_and_ids_are_unique(self):
        registry = ExperimentRegistry(self.root / "index.jsonl")
        record = {"experiment_id": "E0001", "status": "success"}
        registry.append(record)
        self.assertEqual(list(registry.records()), [record])
        with self.assertRaises(RegistryError):
            registry.append(record)


if __name__ == "__main__":
    unittest.main()
