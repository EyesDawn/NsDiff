#!/usr/bin/env python3
"""Batch re-evaluate selected LS-Flow/iReflow checkpoints with EnergyScore.

The script deliberately never uses validation/test results to choose a model.  It
selects the nss5 Stage-3 checkpoint whose first ``output.log`` line records
training seed 2025, falling back to seed 2020 only when the former does not
exist for a dataset.

Examples
--------
    bash scripts/iReflow/run_reevaluate_energy_score.sh
    bash scripts/iReflow/run_reevaluate_energy_score.sh \\
        --gpu-wait-timeout-minutes 120
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


# Workers are launched as a Python file, which otherwise makes ``src/analysis``
# their import root.  Always expose the repository root before importing model
# code, independently of the shell launch location.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Traffic",
    "Weather",
    "SolarEnergy",
)
PREFERRED_SEED = 2025
FALLBACK_SEED = 2020
EXPECTED_RUN = "nss5_temp1.0_invtransFalse"
# GPU 1 is intentionally excluded: it is known to be unreliable on this host.
DISABLED_GPU_IDS = frozenset({1})
FIRST_LOG_LINE = re.compile(
    r"^\[[^\]]+\] - run : nss5_temp1\.0_invtransFalse in seed: (?P<seed>\d+)\s*$"
)


@dataclass(frozen=True)
class EvaluationTask:
    dataset: str
    checkpoint: str
    args_json: str
    output_log: str
    checkpoint_seed: int
    log_first_line: str


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so a partial worker result is never consumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().rstrip("\r\n")


def select_checkpoint_for_dataset(
    runs_root: Path,
    dataset: str,
    preferred_seed: int = PREFERRED_SEED,
    fallback_seed: int = FALLBACK_SEED,
) -> EvaluationTask:
    """Select exactly one matching nss5 Stage-3 run without metric-based choice."""
    dataset_root = runs_root / dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing iReflow run directory for {dataset}: {dataset_root}")

    candidates: dict[int, list[EvaluationTask]] = {
        preferred_seed: [],
        fallback_seed: [],
    }
    for output_log in sorted(dataset_root.glob("**/train_mode_1/output.log")):
        line = first_line(output_log)
        match = FIRST_LOG_LINE.fullmatch(line)
        if match is None:
            continue
        seed = int(match.group("seed"))
        if seed not in candidates:
            continue

        run_dir = output_log.parent
        checkpoint = run_dir / "best_model.pth"
        args_json = run_dir / "args.json"
        if not checkpoint.is_file() or not args_json.is_file():
            continue
        candidates[seed].append(
            EvaluationTask(
                dataset=dataset,
                checkpoint=str(checkpoint.resolve()),
                args_json=str(args_json.resolve()),
                output_log=str(output_log.resolve()),
                checkpoint_seed=seed,
                log_first_line=line,
            )
        )

    for seed in (preferred_seed, fallback_seed):
        matches = candidates[seed]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            locations = "\n".join(f"  - {item.checkpoint}" for item in matches)
            raise RuntimeError(
                f"Ambiguous {dataset} checkpoint selection for seed={seed}; "
                f"expected one {EXPECTED_RUN} Stage-3 run, found:\n{locations}"
            )

    raise FileNotFoundError(
        f"No {EXPECTED_RUN} Stage-3 checkpoint for {dataset} with first-line "
        f"seed={preferred_seed} or fallback seed={fallback_seed} under {dataset_root}"
    )


def discover_tasks(runs_root: Path) -> list[EvaluationTask]:
    return [select_checkpoint_for_dataset(runs_root, dataset) for dataset in DATASETS]


def visible_gpu_indices() -> set[int] | None:
    """Return physical GPU ids visible to this launcher, or None for all GPUs."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None or configured.strip() in {"", "-1"}:
        return None
    ids: set[int] = set()
    for raw_value in configured.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        if not raw_value.isdigit():
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must contain numeric physical GPU ids for this launcher; "
                f"received {configured!r}."
            )
        ids.add(int(raw_value))
    return ids


def exclude_disabled_gpus(memory_by_gpu: dict[int, int]) -> dict[int, int]:
    """Remove host GPUs that must never receive an evaluation worker."""
    return {
        gpu_id: free_mib
        for gpu_id, free_mib in memory_by_gpu.items()
        if gpu_id not in DISABLED_GPU_IDS
    }


def query_gpu_free_memory() -> dict[int, int]:
    """Return physical GPU index -> currently free memory in MiB."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("nvidia-smi is required for GPU scheduling but was not found.") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"nvidia-smi failed while checking available memory: {detail}") from error

    allowed = visible_gpu_indices()
    memory_by_gpu: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"Unexpected nvidia-smi output line: {line!r}")
        gpu_id, free_mib = (int(field) for field in fields)
        if allowed is None or gpu_id in allowed:
            memory_by_gpu[gpu_id] = free_mib
    memory_by_gpu = exclude_disabled_gpus(memory_by_gpu)
    if not memory_by_gpu:
        raise RuntimeError(
            "No usable CUDA GPUs are visible to the evaluation launcher "
            f"after excluding disabled GPUs: {sorted(DISABLED_GPU_IDS)}."
        )
    return memory_by_gpu


def eligible_gpus(
    free_memory_mib: dict[int, int],
    active_workers: dict[int, int],
    min_free_memory_mib: int,
    max_workers_per_gpu: int,
) -> list[int]:
    """Order eligible GPUs by fewest active workers then greatest free memory."""
    return sorted(
        (
            gpu_id
            for gpu_id, free_mib in free_memory_mib.items()
            if free_mib >= min_free_memory_mib
            and active_workers.get(gpu_id, 0) < max_workers_per_gpu
        ),
        key=lambda gpu_id: (active_workers.get(gpu_id, 0), -free_memory_mib[gpu_id], gpu_id),
    )


def _load_iReflow_config(task: EvaluationTask, device: str) -> dict[str, Any]:
    with Path(task.args_json).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    # Training-only and launcher-specific fields are either irrelevant during
    # inference or must be controlled by the re-evaluator.
    config.update(
        {
            "device": device,
            "wandb_project": None,
            "is_training": 0,
        }
    )
    return config


def evaluate_task(task: EvaluationTask, physical_gpu_id: int) -> dict[str, Any]:
    """Run a single, EnergyScore-only test pass in an isolated GPU process."""
    import torch
    from torch_timeseries.utils.reproduce import reproducible

    from src.experiments.iReflow import iReflowExp
    from src.metrics import EnergyScore

    started_at = time.time()
    config = _load_iReflow_config(task, device="cuda:0")
    experiment = iReflowExp(**config)

    # Rebuild the test loader/model from the checkpoint's saved configuration.
    # This avoids calling run(), which redirects logging into the original
    # training run directory and computes unrelated metrics.
    reproducible(task.checkpoint_seed)
    experiment._init_data_loader(shuffle=False, fast_test=False, fast_val=False)
    experiment._init_model()
    checkpoint_state = torch.load(task.checkpoint, map_location=experiment.device, weights_only=True)
    experiment.model.load_state_dict(checkpoint_state)
    experiment.model.eval()

    metric = EnergyScore().to("cpu")
    with torch.no_grad():
        for batch in experiment.test_loader:
            batch_x, batch_y, _origin_x, origin_y, batch_x_date_enc, _batch_y_date_enc = batch
            for (
                _start,
                _end,
                micro_x,
                micro_y,
                micro_origin_y,
                micro_x_date_enc,
                _micro_y_date_enc,
            ) in experiment._iter_eval_micro_batches(
                batch_x,
                batch_y,
                origin_y,
                batch_x_date_enc,
                _batch_y_date_enc,
            ):
                samples, _y_hat, _sigma, _z_samples, _x_samples = experiment.model.forecast(
                    x_enc=micro_x.to(experiment.device).float(),
                    x_mark_enc=micro_x_date_enc.to(experiment.device).float(),
                    num_samples=experiment.num_samples,
                    temperature=experiment.temperature,
                    return_trajs=False,
                )
                predictions = samples.permute(0, 2, 3, 1).contiguous()
                targets = micro_y.to(experiment.device).float()
                if experiment.invtrans_loss:
                    predictions = experiment.scaler.inverse_transform(predictions)
                    targets = micro_origin_y.to(experiment.device).float()
                metric.update(predictions.detach().cpu(), targets.detach().cpu())
                del samples, predictions, targets

    energy_score = float(metric.compute())
    elapsed_seconds = time.time() - started_at
    return {
        "status": "completed",
        "dataset": task.dataset,
        "energy_score": energy_score,
        "checkpoint": task.checkpoint,
        "checkpoint_seed": task.checkpoint_seed,
        "log_first_line": task.log_first_line,
        "physical_gpu_id": physical_gpu_id,
        "ode_sampling_steps": int(experiment.num_sampling_steps),
        "num_samples": int(experiment.num_samples),
        "elapsed_seconds": elapsed_seconds,
    }


def worker_main(task_path: Path, result_path: Path, physical_gpu_id: int) -> int:
    task = EvaluationTask(**json.loads(task_path.read_text(encoding="utf-8")))
    try:
        result = evaluate_task(task, physical_gpu_id)
    except Exception as error:  # Persist the error so the summary remains complete.
        result = {
            "status": "failed",
            "dataset": task.dataset,
            "checkpoint": task.checkpoint,
            "checkpoint_seed": task.checkpoint_seed,
            "log_first_line": task.log_first_line,
            "physical_gpu_id": physical_gpu_id,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        write_json(result_path, result)
        print(result["traceback"], file=sys.stderr, flush=True)
        return 1
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def read_worker_result(result_path: Path, task: EvaluationTask, gpu_id: int, return_code: int) -> dict[str, Any]:
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "status": "failed",
        "dataset": task.dataset,
        "checkpoint": task.checkpoint,
        "checkpoint_seed": task.checkpoint_seed,
        "log_first_line": task.log_first_line,
        "physical_gpu_id": gpu_id,
        "error": f"Worker exited with code {return_code} without writing a result file.",
    }


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Evaluation output directory is non-empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "tasks").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)


def write_summary(output_dir: Path, results: Iterable[dict[str, Any]]) -> None:
    results_by_dataset = {result["dataset"]: result for result in results}
    ordered_results = [results_by_dataset[dataset] for dataset in DATASETS if dataset in results_by_dataset]
    write_json(output_dir / "summary.json", ordered_results)

    fields = [
        "dataset",
        "status",
        "energy_score",
        "checkpoint_seed",
        "checkpoint",
        "physical_gpu_id",
        "ode_sampling_steps",
        "num_samples",
        "elapsed_seconds",
        "log_first_line",
        "error",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered_results)


def launch_workers(
    tasks: list[EvaluationTask],
    output_dir: Path,
    min_free_memory_mib: int,
    max_workers_per_gpu: int,
    poll_seconds: float,
    gpu_wait_timeout_seconds: float | None,
) -> list[dict[str, Any]]:
    pending = list(tasks)
    running: dict[int, tuple[subprocess.Popen[Any], EvaluationTask, int, Any]] = {}
    active_workers: dict[int, int] = {}
    results: list[dict[str, Any]] = []
    wait_started = time.monotonic()
    launcher_log = (output_dir / "launcher.log").open("a", encoding="utf-8")

    try:
        while pending or running:
            completed_processes = [
                pid_and_entry for pid_and_entry in running.items() if pid_and_entry[1][0].poll() is not None
            ]
            for pid, (process, task, gpu_id, log_handle) in completed_processes:
                return_code = process.wait()
                log_handle.close()
                del running[pid]
                active_workers[gpu_id] -= 1
                result_path = output_dir / "results" / f"{task.dataset}.json"
                result = read_worker_result(result_path, task, gpu_id, return_code)
                results.append(result)
                print(
                    f"completed dataset={task.dataset} gpu={gpu_id} status={result['status']}",
                    file=launcher_log,
                    flush=True,
                )

            if pending:
                free_memory = query_gpu_free_memory()
                candidates = eligible_gpus(
                    free_memory,
                    active_workers,
                    min_free_memory_mib,
                    max_workers_per_gpu,
                )
                while pending and candidates:
                    gpu_id = candidates.pop(0)
                    task = pending.pop(0)
                    task_path = output_dir / "tasks" / f"{task.dataset}.json"
                    result_path = output_dir / "results" / f"{task.dataset}.json"
                    write_json(task_path, asdict(task))
                    environment = dict(os.environ)
                    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    log_handle = (output_dir / "logs" / f"{task.dataset}.log").open("w", encoding="utf-8")
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        "--task",
                        str(task_path),
                        "--result",
                        str(result_path),
                        "--physical-gpu-id",
                        str(gpu_id),
                    ]
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    running[process.pid] = (process, task, gpu_id, log_handle)
                    active_workers[gpu_id] = active_workers.get(gpu_id, 0) + 1
                    wait_started = time.monotonic()
                    print(
                        f"started dataset={task.dataset} gpu={gpu_id} free_mib={free_memory[gpu_id]} "
                        f"active_workers={active_workers[gpu_id]}",
                        file=launcher_log,
                        flush=True,
                    )
                    candidates = eligible_gpus(
                        free_memory,
                        active_workers,
                        min_free_memory_mib,
                        max_workers_per_gpu,
                    )

                if not running and pending:
                    waited_seconds = time.monotonic() - wait_started
                    if gpu_wait_timeout_seconds is not None and waited_seconds >= gpu_wait_timeout_seconds:
                        remaining = ", ".join(task.dataset for task in pending)
                        raise TimeoutError(
                            f"No eligible GPU (free memory >= {min_free_memory_mib} MiB) for {waited_seconds:.0f}s; "
                            f"pending datasets: {remaining}"
                        )
                    print(
                        f"waiting for GPU with >= {min_free_memory_mib} MiB free; "
                        f"observed free MiB: {free_memory}",
                        file=launcher_log,
                        flush=True,
                    )

            if pending or running:
                time.sleep(poll_seconds)
    finally:
        # A scheduler error (for example, a lost nvidia-smi connection) must
        # not leave orphaned workers consuming GPUs after the launcher exits.
        for process, _task, _gpu_id, log_handle in running.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_handle.close()
        launcher_log.close()

    return results


def command_main(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    prepare_output_directory(output_dir, args.overwrite)
    tasks = discover_tasks(runs_root)
    write_json(output_dir / "selected_checkpoints.json", [asdict(task) for task in tasks])

    try:
        results = launch_workers(
            tasks,
            output_dir,
            min_free_memory_mib=args.min_free_memory_mib,
            max_workers_per_gpu=args.max_workers_per_gpu,
            poll_seconds=args.poll_seconds,
            gpu_wait_timeout_seconds=(
                None if args.gpu_wait_timeout_minutes is None else args.gpu_wait_timeout_minutes * 60
            ),
        )
    except Exception as error:
        partial_results: list[dict[str, Any]] = []
        for task in tasks:
            result_path = output_dir / "results" / f"{task.dataset}.json"
            if result_path.is_file():
                with result_path.open("r", encoding="utf-8") as handle:
                    partial_results.append(json.load(handle))
            else:
                partial_results.append(
                    {
                        "status": "not_completed",
                        "dataset": task.dataset,
                        "checkpoint": task.checkpoint,
                        "checkpoint_seed": task.checkpoint_seed,
                        "log_first_line": task.log_first_line,
                        "error": f"Launcher stopped: {type(error).__name__}: {error}",
                    }
                )
        write_summary(output_dir, partial_results)
        write_json(
            output_dir / "launcher_failure.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()},
        )
        raise

    write_summary(output_dir, results)
    failed = [result for result in results if result["status"] != "completed"]
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--task", required=True, type=Path)
    worker.add_argument("--result", required=True, type=Path)
    worker.add_argument("--physical-gpu-id", required=True, type=int)

    parser.add_argument("--runs-root", default="results/runs/iReflow")
    parser.add_argument(
        "--output-dir",
        default="results/evaluations/ireflow_energy_score_seed2025",
    )
    parser.add_argument("--min-free-memory-mib", type=int, default=6144)
    parser.add_argument("--max-workers-per-gpu", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--gpu-wait-timeout-minutes",
        type=float,
        default=None,
        help="Fail after this many minutes with no eligible GPU; default waits indefinitely.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "worker":
        return worker_main(args.task, args.result, args.physical_gpu_id)
    if args.min_free_memory_mib < 1:
        parser.error("--min-free-memory-mib must be positive")
    if args.max_workers_per_gpu < 1:
        parser.error("--max-workers-per-gpu must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return command_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
