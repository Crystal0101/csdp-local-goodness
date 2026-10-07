"""Portable orchestration; the hash-locked CSDP training equations stay untouched."""
from pathlib import Path
import argparse
import importlib.util
import os
import sys

from .config import ROOT, load_config, read_json, verify_sources
from .results import compare_first_pair, export_training_csv, SavedResults
from .utils import display_path, fresh_directory, run_logged


def check_mechanisms() -> Path:
    output = fresh_directory("mechanism")
    for name, helper in (
        ("primary", "check_grouped_delivery_mechanism.py"),
        ("supplement", "check_grouped_supplement_mechanism.py"),
    ):
        run_logged(
            [ROOT / "src" / helper, "--study", ROOT / "studies" / name,
             "--output", output / name], output / f"{name}.log",
            python=os.environ.get("CSDP_PYTHON", sys.executable),
        )
        if not read_json(output / name / "result.json")["passed"]:
            raise ValueError(f"{name} mechanism checks failed")
    print("Source checks passed: original G1, derivatives and group support.")
    return output


def reproduce_first_pair(saved: SavedResults) -> Path:
    output = fresh_directory("notebook_pair")
    run_logged(
        [ROOT / "run.py", "--output", output], output.parent / f"{output.name}.log",
        python=os.environ.get("CSDP_PYTHON", sys.executable),
    )
    compare_first_pair(output / "training", saved)
    print("Both fresh models match all five saved epochs, including probability hashes.")
    print(f"Fresh records and logs: {display_path(output)}")
    return output


def run_primary_worker(output: Path, seed: int, group: int) -> None:
    config = load_config("primary")
    config.validate_group(group)
    if seed not in config.seeds:
        raise ValueError("Seed is outside the frozen primary study")
    runner = ROOT / "src/run_grouped_csdp_replication.py"
    spec = importlib.util.spec_from_file_location("frozen_primary", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.O = output / "training"
    module.S = module.O / "source"
    module.DATA = output / "data"
    module.verify()
    # The frozen worker splits JAX keys before initialization; do not reseed encoders.
    module.worker(group, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["primary", "supplement"], default="primary")
    parser.add_argument("--all", action="store_true", help="Run all frozen seeds")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_group", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = (args.output or fresh_directory(args.study)).resolve()
    config = load_config(args.study)
    verify_sources(config)
    if args._seed is not None or args._group is not None:
        if args.study != "primary" or args._seed is None or args._group is None:
            raise ValueError("Internal workers require a primary seed and group")
        run_primary_worker(output, args._seed, args._group)
        return
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    output.mkdir(parents=True)
    source = ROOT / "src"
    primary = ROOT / "studies/primary"
    run_logged(
        [source / "prepare_grouped_data.py", "--output", output / "data",
         "--study", primary], output / "prepare.log",
    )
    mechanism = ("check_grouped_delivery_mechanism.py" if args.study == "primary"
                 else "check_grouped_supplement_mechanism.py")
    run_logged(
        [source / mechanism, "--study", config.directory, "--output", output / "mechanism"],
        output / "mechanism.log",
    )
    if args.study == "supplement":
        mode = ("--prepare-only" if args.prepare_only else
                "--all" if args.all else "--first-pair")
        run_logged(
            [source / "reproduce_grouped_supplement.py", "--study", config.directory,
             "--original-study", primary, "--runner", source / "run_grouped_supplement.py",
             "--data", output / "data", "--output", output / "training", mode],
            output / "training.log",
        )
    else:
        command = [
            source / "reproduce_grouped_pair.py", "--study", config.directory,
            "--runner", source / "run_grouped_csdp_replication.py",
            "--data", output / "data", "--output", output / "training",
        ]
        if args.all or args.prepare_only:
            command.append("--prepare-only")
        run_logged(command, output / "training.log")
        if args.all and not args.prepare_only:
            # Resolver contexts are mutable; each trial needs its own process.
            for seed in config.seeds:
                for group in config.groups:
                    run_logged(
                        [ROOT / "run.py", "--output", output,
                         "--_seed", seed, "--_group", group],
                        output / f"G{group}_{seed}.log",
                    )
    if args.prepare_only:
        print(f"Prepared inputs and source checks; no training: {display_path(output)}")
        return
    expected = ({(group, seed) for group in config.groups for seed in config.seeds}
                if args.all else {(group, config.seeds[0]) for group in (1, 8)})
    actual = set()
    for file in (output / "training/runs").glob("*/result.json"):
        record = read_json(file)
        actual.add((record["group"], record["seed"]))
    if actual != expected:
        raise ValueError("Fresh output is missing a planned run")
    export_training_csv(output / "training", args.study)
    print(f"Completed {len(expected)} fixed trials: {display_path(output)}")
    print("Saved all epochs in training/epochs.csv and final metrics in training/trial_metrics.csv.")
