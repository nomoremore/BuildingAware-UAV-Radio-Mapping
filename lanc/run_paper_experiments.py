from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


STABILITY_SEEDS = (42, 2024, 3407)
STABILITY_STRATEGIES = ("random", "uniform_grid", "weak_only", "proposed", "proposed_v2_j")
BUDGET_SETTINGS = ((0.02, 0.01), (0.05, 0.02), (0.10, 0.02))
BUDGET_STRATEGIES = ("random", "proposed_v2_j")


@dataclass(frozen=True)
class BatchConfig:
    suite: str
    source_run: Path
    target_farm_root: Path
    target_scene_id: int
    output_root: Path = Path("runs")
    rounds: int = 5
    epochs: int = 8
    stride: int = 16
    batch_size: int = 2048
    learning_rate: float = 1e-4
    frequency: str = "freq35"
    antenna_pattern: str = "pattern_120"
    seeds: tuple[int, ...] = STABILITY_SEEDS
    budget_seed: int = 42
    cpu: bool = False


@dataclass(frozen=True)
class PaperExperimentSpec:
    suite: str
    label: str
    source_run: Path
    target_farm_root: Path
    target_scene_id: int
    seed: int
    initial_ratio: float
    budget_ratio: float
    strategies: tuple[str, ...]
    rounds: int
    epochs: int
    stride: int
    batch_size: int
    learning_rate: float
    frequency: str
    antenna_pattern: str
    cpu: bool
    active_output_root: Path = Path("runs")

    @property
    def config_key(self) -> str:
        # 配置键只描述实验条件，不包含时间戳输出目录，这样 --skip-existing 可以跨 batch 识别重复实验。
        payload = {
            "suite": self.suite,
            "source_run": str(self.source_run),
            "target_farm_root": str(self.target_farm_root),
            "target_scene_id": self.target_scene_id,
            "seed": self.seed,
            "initial_ratio": self.initial_ratio,
            "budget_ratio": self.budget_ratio,
            "strategies": list(self.strategies),
            "rounds": self.rounds,
            "epochs": self.epochs,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "frequency": self.frequency,
            "antenna_pattern": self.antenna_pattern,
            "cpu": self.cpu,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final paper experiment batches for FARM-adapted LANC.")
    parser.add_argument("--suite", choices=("stability", "budget", "all"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-farm-root", type=Path, required=True)
    parser.add_argument("--target-scene-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--frequency", default="freq35")
    parser.add_argument("--antenna-pattern", default="pattern_120")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without creating training runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip completed configs found in previous batches.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BatchConfig(
        suite=args.suite,
        source_run=args.source_run,
        target_farm_root=args.target_farm_root,
        target_scene_id=args.target_scene_id,
        output_root=args.output_root,
        rounds=args.rounds,
        epochs=args.epochs,
        stride=args.stride,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        frequency=args.frequency,
        antenna_pattern=args.antenna_pattern,
        cpu=args.cpu,
    )
    specs = build_experiment_specs(config)
    if args.skip_existing:
        specs = filter_existing_specs(specs, load_existing_completed_keys(args.output_root))

    if args.dry_run:
        print(f"Dry run: {len(specs)} experiment(s) would be executed.")
        for spec in specs:
            print(" ".join(command_for_spec(spec, sys.executable)))
        return

    run_batch(specs, args.output_root)


def build_experiment_specs(config: BatchConfig) -> list[PaperExperimentSpec]:
    specs: list[PaperExperimentSpec] = []
    suites = ("stability", "budget") if config.suite == "all" else (config.suite,)
    for suite in suites:
        if suite == "stability":
            for seed in config.seeds:
                specs.append(
                    _make_spec(
                        config=config,
                        suite=suite,
                        label=f"stability_seed{seed}",
                        seed=seed,
                        initial_ratio=0.05,
                        budget_ratio=0.02,
                        strategies=STABILITY_STRATEGIES,
                    )
                )
        elif suite == "budget":
            for initial_ratio, budget_ratio in BUDGET_SETTINGS:
                specs.append(
                    _make_spec(
                        config=config,
                        suite=suite,
                        label=f"budget_i{_ratio_label(initial_ratio)}_b{_ratio_label(budget_ratio)}",
                        seed=config.budget_seed,
                        initial_ratio=initial_ratio,
                        budget_ratio=budget_ratio,
                        strategies=BUDGET_STRATEGIES,
                    )
                )
        else:
            raise ValueError(f"Unknown paper experiment suite: {suite}")
    return specs


def command_for_spec(spec: PaperExperimentSpec, python_executable: str | Path | None = None) -> list[str]:
    python = str(python_executable or sys.executable)
    command = [
        python,
        "-m",
        "lanc.run_active_experiment",
        "--source-run",
        str(spec.source_run),
        "--target-farm-root",
        str(spec.target_farm_root),
        "--target-scene-id",
        str(spec.target_scene_id),
        "--frequency",
        spec.frequency,
        "--antenna-pattern",
        spec.antenna_pattern,
        "--stride",
        str(spec.stride),
        "--rounds",
        str(spec.rounds),
        "--initial-ratio",
        str(spec.initial_ratio),
        "--budget-ratio",
        str(spec.budget_ratio),
        "--seed",
        str(spec.seed),
        "--epochs",
        str(spec.epochs),
        "--batch-size",
        str(spec.batch_size),
        "--learning-rate",
        str(spec.learning_rate),
        "--output-root",
        str(spec.active_output_root),
    ]
    if spec.cpu:
        command.append("--cpu")
    command.extend(["--strategies", *spec.strategies])
    return command


def filter_existing_specs(specs: list[PaperExperimentSpec], existing_keys: set[str]) -> list[PaperExperimentSpec]:
    return [spec for spec in specs if spec.config_key not in existing_keys]


def load_existing_completed_keys(output_root: Path) -> set[str]:
    keys: set[str] = set()
    for manifest_path in output_root.glob("paper_batch_*/batch_manifest.csv"):
        try:
            manifest = pd.read_csv(manifest_path)
        except Exception:
            continue
        if "config_key" not in manifest.columns or "status" not in manifest.columns:
            continue
        completed = manifest[manifest["status"].astype(str) == "completed"]
        keys.update(completed["config_key"].dropna().astype(str))
    return keys


def run_batch(specs: list[PaperExperimentSpec], output_root: Path) -> Path:
    batch_dir = output_root / f"paper_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    active_output_root = batch_dir / "active_runs"
    log_dir = batch_dir / "logs"
    active_output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    manifest_path = batch_dir / "batch_manifest.csv"
    if not specs:
        pd.DataFrame(columns=_manifest_columns()).to_csv(manifest_path, index=False, encoding="utf-8-sig")
        _write_batch_manifest_json(batch_dir, specs, manifest_path)
        print(f"No experiments to run. Empty batch manifest: {manifest_path.resolve()}")
        return batch_dir

    for spec in specs:
        spec = _spec_with_active_output_root(spec, active_output_root)
        row = _base_manifest_row(spec)
        row["started_at"] = datetime.now().isoformat(timespec="seconds")
        log_path = log_dir / f"{spec.label}.log"
        row["log_path"] = str(log_path)
        command = command_for_spec(spec, sys.executable)
        row["command"] = " ".join(command)
        print(f"Running {spec.label}: {row['command']}")
        returncode, output = _run_command(command, log_path)
        row["returncode"] = returncode
        row["ended_at"] = datetime.now().isoformat(timespec="seconds")
        if returncode == 0:
            row["status"] = "completed"
            row["run_dir"] = _parse_run_dir(output)
            row["error_message"] = ""
        else:
            row["status"] = "failed"
            row["run_dir"] = _parse_run_dir(output)
            row["error_message"] = _last_nonempty_line(output)
        rows.append(row)
        pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")

    _write_batch_manifest_json(batch_dir, specs, manifest_path)
    print(f"Batch directory: {batch_dir.resolve()}")
    print(f"Batch manifest: {manifest_path.resolve()}")
    return batch_dir


def _make_spec(
    config: BatchConfig,
    suite: str,
    label: str,
    seed: int,
    initial_ratio: float,
    budget_ratio: float,
    strategies: tuple[str, ...],
) -> PaperExperimentSpec:
    return PaperExperimentSpec(
        suite=suite,
        label=label,
        source_run=config.source_run,
        target_farm_root=config.target_farm_root,
        target_scene_id=config.target_scene_id,
        seed=seed,
        initial_ratio=initial_ratio,
        budget_ratio=budget_ratio,
        strategies=strategies,
        rounds=config.rounds,
        epochs=config.epochs,
        stride=config.stride,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        frequency=config.frequency,
        antenna_pattern=config.antenna_pattern,
        cpu=config.cpu,
    )


def _spec_with_active_output_root(spec: PaperExperimentSpec, active_output_root: Path) -> PaperExperimentSpec:
    return PaperExperimentSpec(
        suite=spec.suite,
        label=spec.label,
        source_run=spec.source_run,
        target_farm_root=spec.target_farm_root,
        target_scene_id=spec.target_scene_id,
        seed=spec.seed,
        initial_ratio=spec.initial_ratio,
        budget_ratio=spec.budget_ratio,
        strategies=spec.strategies,
        rounds=spec.rounds,
        epochs=spec.epochs,
        stride=spec.stride,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        frequency=spec.frequency,
        antenna_pattern=spec.antenna_pattern,
        cpu=spec.cpu,
        active_output_root=active_output_root,
    )


def _base_manifest_row(spec: PaperExperimentSpec) -> dict[str, object]:
    return {
        "suite": spec.suite,
        "label": spec.label,
        "config_key": spec.config_key,
        "status": "pending",
        "seed": spec.seed,
        "initial_ratio": spec.initial_ratio,
        "budget_ratio": spec.budget_ratio,
        "strategies": " ".join(spec.strategies),
        "rounds": spec.rounds,
        "epochs": spec.epochs,
        "stride": spec.stride,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "source_run": str(spec.source_run),
        "target_farm_root": str(spec.target_farm_root),
        "target_scene_id": spec.target_scene_id,
    }


def _manifest_columns() -> list[str]:
    return [
        "suite",
        "label",
        "config_key",
        "status",
        "seed",
        "initial_ratio",
        "budget_ratio",
        "strategies",
        "rounds",
        "epochs",
        "stride",
        "batch_size",
        "learning_rate",
        "source_run",
        "target_farm_root",
        "target_scene_id",
        "started_at",
        "ended_at",
        "command",
        "run_dir",
        "log_path",
        "returncode",
        "error_message",
    ]


def _run_command(command: list[str], log_path: Path) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)
    return process.wait(), "".join(output_lines)


def _parse_run_dir(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Run directory:"):
            return line.split(":", 1)[1].strip()
    return ""


def _last_nonempty_line(output: str) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return ""


def _write_batch_manifest_json(batch_dir: Path, specs: list[PaperExperimentSpec], manifest_path: Path) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_csv": str(manifest_path),
        "experiment_count": len(specs),
        "note": "Each completed row points to one active sampling run directory.",
    }
    (batch_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ratio_label(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
