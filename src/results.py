"""Validate compact predictions and export fresh runs without selecting outcomes."""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np

from .config import ROOT, load_config, read_json
from .utils import probability_digest


@dataclass
class SavedResults:
    accuracy: dict[tuple[str, int, int, int], float]
    rows: dict[tuple[str, int, int, int], dict]

    def values(self, cohort: str, group: int, epoch: int) -> np.ndarray:
        config = load_config(cohort)
        config.validate_group(group)
        return np.array([
            self.accuracy[cohort, group, seed, epoch] for seed in config.seeds
        ])

    def curves(self, cohort: str, group: int) -> np.ndarray:
        config = load_config(cohort)
        return np.array([
            self.values(cohort, group, epoch)
            for epoch in range(1, config.epochs[group] + 1)
        ]).T


def load_saved_results(directory: Path | None = None) -> SavedResults:
    directory = directory or ROOT / "results"
    labels = np.asarray(read_json(directory / "labels.json"))
    if labels.shape != (359,):
        raise ValueError("Expected the frozen 359-image validation split")
    accuracy, records = {}, {}
    with (directory / "epochs.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            key = (row["cohort"], int(row["group"]), int(row["seed"]), int(row["epoch"]))
            if key in records:
                raise ValueError(f"Duplicate epoch record: {key}")
            prediction = np.fromstring(row["predictions"], dtype=int, sep=" ")
            if prediction.shape != labels.shape or not np.isin(prediction, range(10)).all():
                raise ValueError(f"Invalid predictions: {key}")
            measured = float(np.mean(prediction == labels))
            if not np.isfinite(float(row["accuracy"])):
                raise ValueError(f"Nonfinite saved accuracy: {key}")
            if abs(measured - float(row["accuracy"])) > 1e-12:
                raise ValueError(f"Prediction/accuracy mismatch: {key}")
            accuracy[key], records[key] = 100 * measured, row
    expected = set()
    for cohort in ("primary", "supplement"):
        config = load_config(cohort)
        expected.update(
            (cohort, group, seed, epoch)
            for group in config.groups for seed in config.seeds
            for epoch in range(1, config.epochs[group] + 1)
        )
    if set(records) != expected:
        raise ValueError("Missing or unexpected seed/group/epoch records")
    verify_pairing(read_json(directory / "pairing.json"))
    return SavedResults(accuracy, records)


def verify_pairing(pairing: dict) -> None:
    for cohort in ("primary", "supplement"):
        config = load_config(cohort)
        for seed in config.seeds:
            reference = pairing[f"{cohort}/G1_{seed}"]
            for group in config.groups:
                row = pairing[f"{cohort}/G{group}_{seed}"]
                if not row["all_finite_checks_passed"] or not row["inference_labels_zero"]:
                    raise ValueError("A saved run failed its integrity checks")
                for field in ("initial_weight_hashes", "weight_count"):
                    if row[field] != reference[field]:
                        raise ValueError(f"Paired {field} mismatch")
                count = config.epochs[group]
                if len(row["stream_by_epoch"]) != count:
                    raise ValueError("Incomplete random-stream audit")
                if row["stream_by_epoch"] != reference["stream_by_epoch"][:count]:
                    raise ValueError("Paired random streams differ")
                if cohort == "supplement":
                    if row["initial_encoder_keys"] != reference["initial_encoder_keys"]:
                        raise ValueError("Initial encoder keys differ")
                    if len(row["encoder_keys_by_epoch"]) != count:
                        raise ValueError("Incomplete encoder-key audit")
                    if row["encoder_keys_by_epoch"] != reference["encoder_keys_by_epoch"][:count]:
                        raise ValueError("Paired encoder-key streams differ")


def export_training_csv(training: Path, cohort: str) -> tuple[Path, Path]:
    """Save every completed run; the caller checks the planned seed/group set."""
    epoch_rows, trial_rows = [], []
    config = load_config(cohort)
    for file in sorted((training / "runs").glob("*/result.json")):
        run = read_json(file)
        group, seed = run["group"], run["seed"]
        config.validate_group(group)
        if seed not in config.seeds:
            raise ValueError("Unexpected seed in fresh output")
        budget = config.epochs[group]
        if len(run["epoch_results"]) != budget or len(run["curves"]) != budget:
            raise ValueError("Incomplete fresh run")
        expected_epochs = list(range(1, budget + 1))
        if ([row["epoch"] for row in run["epoch_results"]] != expected_epochs
                or [row["epoch"] for row in run["curves"]] != expected_epochs):
            raise ValueError("Fresh epochs must be complete and ordered")
        labels = np.asarray(run["labels"])
        for curve, epoch in zip(run["curves"], run["epoch_results"]):
            prediction = np.asarray(epoch["predictions"])
            probabilities = np.asarray(epoch["probabilities"])
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite fresh probability")
            np.testing.assert_array_equal(probabilities.argmax(1), prediction)
            measured = float(np.mean(prediction == labels))
            if abs(measured - curve["validation_accuracy"]) > 1e-12:
                raise ValueError("Fresh prediction/accuracy mismatch")
            epoch_rows.append({
                "cohort": cohort, "group": group, "seed": seed,
                "epoch": epoch["epoch"], "accuracy": measured,
                "predictions": " ".join(map(str, prediction)),
                "probabilities_sha256": probability_digest(epoch["probabilities"]),
            })
        if abs(measured - run["final_validation_accuracy"]) > 1e-12:
            raise ValueError("Final accuracy differs from the last epoch")
        trial_rows.append({
            "cohort": cohort, "group": group, "seed": seed,
            "epochs": len(run["epoch_results"]),
            "final_accuracy": run["final_validation_accuracy"],
        })
    if not epoch_rows:
        raise ValueError("No completed runs to export")
    for filename, rows in (("epochs.csv", epoch_rows), ("trial_metrics.csv", trial_rows)):
        with (training / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return training / "epochs.csv", training / "trial_metrics.csv"


def compare_first_pair(training: Path, saved: SavedResults) -> None:
    seed = load_config("primary").seeds[0]
    for group in (1, 8):
        run = read_json(training / "runs" / f"G{group}_{seed}" / "result.json")
        for epoch in run["epoch_results"]:
            original = saved.rows["primary", group, seed, epoch["epoch"]]
            if epoch["predictions"] != list(map(int, original["predictions"].split())):
                raise ValueError("Fresh predictions differ; investigate the environment")
            if probability_digest(epoch["probabilities"]) != original["probabilities_sha256"]:
                raise ValueError("Fresh probabilities differ; investigate the environment")
