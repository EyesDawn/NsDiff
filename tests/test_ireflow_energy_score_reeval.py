import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "src/analysis/reevaluate_ireflow_energy_score.py"
SPEC = importlib.util.spec_from_file_location("reevaluate_energy_score", SCRIPT_PATH)
reevaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reevaluate
SPEC.loader.exec_module(reevaluate)


def _make_run(root: Path, dataset: str, seed: int, suffix: str = "run") -> Path:
    run_dir = root / dataset / "w96h1s192" / suffix / "train_mode_1"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").touch()
    (run_dir / "args.json").write_text(json.dumps({}), encoding="utf-8")
    (run_dir / "output.log").write_text(
        f"[2026-01-01 00:00:00] - run : nss5_temp1.0_invtransFalse in seed: {seed}\n",
        encoding="utf-8",
    )
    return run_dir


class IReflowEnergyScoreReevaluationTest(unittest.TestCase):
    def test_checkpoint_selection_prefers_seed_2025_over_seed_2020(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = _make_run(root, "ETTh1", 2020, "fallback")
            preferred = _make_run(root, "ETTh1", 2025, "preferred")

            task = reevaluate.select_checkpoint_for_dataset(root, "ETTh1")

            self.assertEqual(task.checkpoint_seed, 2025)
            self.assertEqual(Path(task.checkpoint).parent, preferred)
            self.assertNotEqual(Path(task.checkpoint).parent, fallback)

    def test_checkpoint_selection_falls_back_to_seed_2020(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = _make_run(root, "ETTm2", 2020, "fallback")

            task = reevaluate.select_checkpoint_for_dataset(root, "ETTm2")

            self.assertEqual(task.checkpoint_seed, 2020)
            self.assertEqual(Path(task.checkpoint).parent, fallback)

    def test_checkpoint_selection_rejects_ambiguous_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_run(root, "Weather", 2025, "first")
            _make_run(root, "Weather", 2025, "second")

            with self.assertRaisesRegex(RuntimeError, "Ambiguous Weather checkpoint"):
                reevaluate.select_checkpoint_for_dataset(root, "Weather")

    def test_gpu_eligibility_uses_memory_threshold_and_worker_limit(self):
        eligible = reevaluate.eligible_gpus(
            free_memory_mib={0: 6144, 1: 6143, 2: 9000},
            active_workers={0: 3, 2: 1},
            min_free_memory_mib=6144,
            max_workers_per_gpu=3,
        )

        self.assertEqual(eligible, [2])

    def test_gpu_1_is_excluded_even_if_it_has_free_memory(self):
        usable = reevaluate.exclude_disabled_gpus({0: 8000, 1: 32000, 2: 8000})

        self.assertEqual(usable, {0: 8000, 2: 8000})
