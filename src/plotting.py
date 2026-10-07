"""Plot fixed saved outcomes; no training and no random jitter or resampling."""
import numpy as np
import matplotlib.pyplot as plt

from .config import load_config
from .results import SavedResults
from .stats import Comparison


def plot_paired_seeds(saved: SavedResults):
    reference = saved.values("primary", 1, 5)
    grouped = saved.values("primary", 8, 5)
    seeds = load_config("primary").seeds
    figure, axes = plt.subplots(1, 2, figsize=(9, 3))
    for first, second in zip(reference, grouped):
        axes[0].plot([0, 1], [first, second], color="0.65", linewidth=.7)
    axes[0].scatter(np.zeros(len(seeds)), reference, s=15, color="black")
    axes[0].scatter(np.ones(len(seeds)), grouped, s=15, color="black")
    axes[0].set_xticks([0, 1], ["G=1", "G=8"])
    axes[0].set_ylabel("Epoch-5 accuracy (%)")
    axes[1].scatter(seeds, grouped - reference, s=15, color="black")
    axes[1].axhline(0, color="0.7")
    axes[1].axhline(-1, color="black", linestyle="--", label="-1 pp margin")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("G8 - G1 (pp)")
    axes[1].ticklabel_format(useOffset=False)
    axes[1].legend()
    figure.tight_layout()
    return figure


def plot_failure_case(saved: SavedResults):
    seeds = load_config("primary").seeds
    differences = saved.values("primary", 8, 5) - saved.values("primary", 1, 5)
    # Select the worst run for diagnosis, never for the aggregate result.
    seed = seeds[int(np.argmin(differences))]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3))
    curves = {}
    for group, style in ((1, "-"), (8, "--")):
        curves[group] = np.array([
            saved.accuracy["primary", group, seed, epoch] for epoch in range(1, 6)
        ])
        axes[0].plot(range(1, 6), curves[group], style, marker="o", label=f"G={group}")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[1].plot(range(1, 6), curves[8] - curves[1], "o-", color="black")
    axes[1].axhline(0, color="0.7")
    axes[1].axhline(-1, color="0.5", linestyle="--")
    axes[1].set_ylabel("G8 - G1 (pp)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(range(1, 6))
    figure.suptitle(f"Seed {seed}: worst final paired loss")
    figure.tight_layout()
    return figure


def plot_supplement(saved: SavedResults, comparisons: dict[str, Comparison]):
    groups = load_config("supplement").groups
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(groups, [saved.values("supplement", group, 5).mean()
                          for group in groups], "o-", color="black")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(groups, [str(group) for group in groups])
    axes[0].set_xlabel("Group count")
    axes[0].set_ylabel("Epoch-5 accuracy (%)")
    for index, group in enumerate(groups[1:]):
        result = comparisons[f"G{group}-G1"]
        axes[1].errorbar(index, result.mean_pp,
                        yerr=[[result.mean_pp - result.lower_pp],
                              [result.upper_pp - result.mean_pp]],
                        fmt="o", color="black", capsize=3)
    axes[1].set_xticks(range(len(groups)-1), [f"{group}-1" for group in groups[1:]])
    axes[1].axhline(0, color="0.7")
    axes[1].axhline(-1, color="0.5", linestyle="--")
    axes[1].set_xlabel("Group contrast")
    axes[1].set_ylabel("Difference (pp)")
    axes[1].set_title("Simultaneous 95% intervals")
    for group, style in ((1, "-"), (8, "--")):
        curves = saved.curves("supplement", group)
        mean, sd = curves.mean(0), curves.std(0, ddof=1)
        epochs = range(1, curves.shape[1]+1)
        axes[2].plot(epochs, mean, style, label=f"G={group}")
        axes[2].fill_between(epochs, mean-sd, mean+sd, alpha=.15)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].legend()
    figure.tight_layout()
    return figure
