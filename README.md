# Making CSDP’s learning signal more local

I used CSDP to explore learning beyond approximations to backpropagation. Its goodness signal uses activity from the whole layer. I replaced that calculation with fixed groups and asked how much accuracy this smaller dependency costs.

> These commits package completed experiments in reviewable steps; they do not reconstruct the original development chronology. Frozen experiment protocols are kept in `studies/`.

**[Open the notebook](demo.ipynb).** It contains the mechanism check, every seed, the extra group-count experiment and a fresh training run. The [two-page appendix](docs/appendix.pdf) is the short version.

## What changed

With eight groups, each instantaneous goodness modulator depends on 32/8 traces instead of 256/64 in the two hidden layers: one eighth of the original fan-in. The loss also changes. Dense recurrent paths and shared signals remain, so I do not call the whole network local or claim an eightfold resource saving.

The author already discusses group/helper mechanisms. My contribution here is the implementation and the measured trade-off, not a new learning principle.

## What happened

| Fixed comparison | G8 minus G1 | Interval, percentage points |
|---|---:|---|
| Original 20 pairs, epoch 5 | -0.153 pp | 95% [-0.765, +0.459] |
| Extra 10 pairs, epoch 5 | -0.641 pp | Simultaneous 95% [-1.683, +0.401] |
| Same extra 10 pairs, epoch 20 | -0.251 pp | Separate 95% [-0.696, +0.194] |

The original mean passed the predeclared -1 pp retention margin. Four of 20 pairs still lost more than 1 pp. Seed 12014 lost 3.90 pp; G8 was ahead at epoch 2 and behind at epoch 3. I have not established why.

I then checked five group counts across ten new seeds: **50 models, not 50 seeds**. The new five-epoch G8 result did not re-establish retention. At 20 epochs the margin was met, but the interval for the change in the gap includes zero. I cannot conclude that longer training narrows the gap. I kept G8 rather than choosing the group with the highest mean.

All runs use the same previously inspected digits validation split. These intervals describe variation across training seeds, not performance on independent data. There is no patient-privacy or clinical result here.

## Reproducibility statement

Seeds and budgets are fixed in the saved protocols. I report all runs, including the losses; I did not choose the best checkpoint or replace seeds. A fresh-kernel run recomputes all 750 saved epoch accuracies and trains the first fixed pair, whose predictions and probability hashes matched the archive in the tested environment. Repeating all 90 models is a separate, optional run. I have not verified a fresh dependency installation or other platforms, so I do not claim universal bitwise reproducibility.

## Known limitations

Grouping reduces instantaneous goodness dependency to one eighth. Accuracy retention depends on the seed cohort and training budget; the extra five-epoch comparison did not re-establish it. Longer training has not been shown to reduce the paired performance gap. I claim neither improved whole-network efficiency nor a novel algorithm. The data split has been used in earlier development.

## Future work

I would first check whether losses come from missing cross-group evidence or from objective scaling and saturation. If the evidence supports it, I would test low-frequency cross-group summaries against fixed grouping and periodic full-layer correction under the same information budget, counting extra state and computation. This would add some nonlocal information back. Its value and novelty still need testing.

## Run it

Python 3.12 on Apple M2 Max CPU was tested. From this directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

This runs source checks and trains the fixed first G1/G8 pair for five epochs. Open `demo.ipynb` with the same environment to inspect the saved comparisons and run the same pair interactively. See [REPRODUCE.md](REPRODUCE.md) for the full studies and notebook execution command.

I kept the author's **JAX/ngclearn** implementation to avoid changing the learning dynamics during a framework port. The supervisor suggested snnTorch as an example, not a requirement; this demo does not import or depend on snnTorch.

## Code map

Analysis, plotting and reproduction orchestration are separate modules in `src/`. The notebook serves as the presentation interface for results and discussion. The hash-locked model and training equations remain in `studies/*/source/` and `src/run_grouped_*.py`; I have not rewritten them for style.

| Read this | To inspect |
|---|---|
| `src/config.py` and `studies/*/protocol.json` | Frozen seed lists, groups, budgets and settings |
| `studies/primary/source/custom/goodnessModCell.py` | The actual grouped goodness calculation |
| `src/trainer.py` | Portable training orchestration and process isolation |
| `src/results.py` | Prediction checks, paired records and CSV export |
| `src/stats.py` | Paired intervals and the four-contrast correction |
| `src/plotting.py` | Deterministic plots from saved results |

The model uses explicit JAX keys, not PyTorch's global seed. The frozen runner splits those keys before initialization; encoder keys then evolve across training and validation. I do not reset them between phases. Plotting adds no jitter or random resampling, so it needs no plotting seed. Numerical or font differences across platforms are still possible.

## Files

- `demo.ipynb`: the experiment and its interpretation.
- `src/`, `studies/`: runner helpers, frozen model source and protocols.
- `results/`: all 750 epoch predictions, per-trial final metrics and pairing checks, in small text files.
- `docs/`: the appendix and a short record of the route I abandoned.

No checkpoints, datasets, environments, ZIPs or third-party paper PDFs are included. Full probability arrays and batch traces remain in my local research archive; rerunning generates fresh records. This compact export checks saved predictions, not omitted probability arrays.

## References

A. G. Ororbia, *Contrastive signal-dependent plasticity: Self-supervised learning in spiking neural circuits*, Science Advances (2024), [doi:10.1126/sciadv.adn6076](https://doi.org/10.1126/sciadv.adn6076). Group/helper discussion: [arXiv:2303.18187v3](https://arxiv.org/abs/2303.18187v3), supplementary text.

The upstream BSD-3-Clause notices are retained with both source snapshots. [Source notes](docs/SOURCE.md) identify the frozen changes and hashes.

## License

BSD-3-Clause; see [LICENSE](LICENSE). The vendored CSDP source retains its upstream copyright and license notices.

## Review and release

Publication was approved by Ning Yang on 7 October 2026. The proposal appendix is identified by the `v1.0-proposal-appendix` tag. The [release checklist](docs/REVIEW_AND_RELEASE.md) separates this evidence from future mechanism work. The tag will not be moved; corrections will be disclosed separately.
