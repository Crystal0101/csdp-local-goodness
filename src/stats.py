"""Paired-seed statistics; never pool images or choose a winning group."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import t

from .config import load_config
from .results import SavedResults


@dataclass(frozen=True)
class Comparison:
    mean_pp: float
    lower_pp: float
    upper_pp: float
    losses_over_one_pp: int
    count: int


def paired_summary(differences: np.ndarray, confidence: float = .95) -> Comparison:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError("At least two paired seed differences are required")
    if not np.isfinite(differences).all() or not 0 < confidence < 1:
        raise ValueError("Finite differences and a confidence level in (0, 1) are required")
    half_width = (t.ppf((1 + confidence) / 2, len(differences) - 1)
                  * differences.std(ddof=1) / np.sqrt(len(differences)))
    mean = float(differences.mean())
    return Comparison(mean, mean - half_width, mean + half_width,
                      int((differences < -1).sum()), len(differences))


def analyse_results(saved: SavedResults) -> dict[str, Comparison]:
    primary = saved.values("primary", 8, 5) - saved.values("primary", 1, 5)
    comparisons = {"primary": paired_summary(primary)}
    groups = load_config("supplement").groups[1:]
    # Four contrasts share a 5% error budget; this is not a pointwise 95% interval.
    adjusted_confidence = 1 - .05 / len(groups)
    for group in groups:
        difference = (saved.values("supplement", group, 5)
                      - saved.values("supplement", 1, 5))
        comparisons[f"G{group}-G1"] = paired_summary(difference, adjusted_confidence)
    long_gap = saved.values("supplement", 8, 20) - saved.values("supplement", 1, 20)
    early_gap = saved.values("supplement", 8, 5) - saved.values("supplement", 1, 5)
    comparisons["epoch20"] = paired_summary(long_gap)
    comparisons["budget_interaction"] = paired_summary(long_gap - early_gap)
    return comparisons


def print_seed_table(saved: SavedResults, cohort: str, epoch: int) -> None:
    config = load_config(cohort)
    groups = [group for group in config.groups if config.epochs[group] >= epoch]
    print(f"{cohort}, epoch {epoch}: accuracy percentages")
    print("seed", *[f"G{group:>7}" for group in groups])
    for seed in config.seeds:
        print(seed, *[f"{saved.accuracy[cohort,group,seed,epoch]:8.3f}" for group in groups])


def print_comparisons(comparisons: dict, names: list[str]) -> None:
    for name in names:
        result = comparisons[name]
        interval_label = "simultaneous 95%" if name.startswith("G") else "95%"
        if name == "budget_interaction":
            interval_label = "descriptive 95%"
        message = (f"{name}: {result.mean_pp:+.3f} pp, {interval_label} interval "
                   f"[{result.lower_pp:+.3f}, {result.upper_pp:+.3f}]")
        if name != "budget_interaction":
            message += f"; losses >1 pp: {result.losses_over_one_pp}/{result.count}"
        print(message)
