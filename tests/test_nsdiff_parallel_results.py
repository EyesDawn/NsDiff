import csv
import tempfile
import unittest
from pathlib import Path

from src.analysis.collect_nsdiff_parallel_results import write_summary


class NsDiffParallelResultsTest(unittest.TestCase):
    def test_collects_last_final_test_result_with_energy_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "status").mkdir()
            (run_dir / "status" / "ETTh2_p48.status").write_text("succeeded\n")
            (run_dir / "logs" / "ETTh2_p48.log").write_text(
                "test_results: {'crps': 99.0}\n"
                "test_results: {'crps': 1.0, 'crps_sum': 2.0, 'qice': 3.0, 'picp': 4.0, "
                "'mse': 5.0, 'mae': 6.0, 'rmse': 7.0, 'es': 8.0}\n"
            )

            output = run_dir / "summary.csv"
            rows = write_summary(run_dir, output, seed=2022)

            self.assertEqual(rows[0]["status"], "succeeded")
            self.assertEqual(rows[0]["es"], 8.0)
            with output.open(newline="") as file:
                saved_rows = list(csv.DictReader(file))
            self.assertEqual(saved_rows[0]["dataset"], "ETTh2")
            self.assertEqual(saved_rows[0]["es"], "8.0")

    def test_failed_job_is_retained_without_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "status").mkdir()
            (run_dir / "status" / "Weather_p96.status").write_text("failed:exit_1\n")

            rows = write_summary(run_dir, run_dir / "summary.csv", seed=2022)

            self.assertEqual(
                rows,
                [
                    {
                        "dataset": "Weather",
                        "pred_len": 96,
                        "seed": 2022,
                        "status": "failed:exit_1",
                        "crps": "",
                        "crps_sum": "",
                        "qice": "",
                        "picp": "",
                        "mse": "",
                        "mae": "",
                        "rmse": "",
                        "es": "",
                        "log_path": str(run_dir / "logs" / "Weather_p96.log"),
                    }
                ],
            )
