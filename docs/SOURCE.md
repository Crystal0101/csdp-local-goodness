# Source and scope

The two `studies/*/source/` trees contain the frozen CSDP implementation used in the experiments, with the upstream LICENSE preserved. `source_manifest.json` gives the SHA-256 of every source file. The frozen runners are kept byte-for-byte because each protocol records its runner hash. The portable wrapper supplies paths; it does not change their update equations.

The adaptation is in `custom/goodnessModCell.py`: fixed equal contiguous groups, scaled logits, and a mean group BCE. G=1 retains the original branch. The supplement changes only the allowed group-count guard and its error message; see its `source_change_record.json`.

Use `run.py` as the portable entry point. Frozen records and workers are evidence, not a second user-facing configuration API.

The upstream author-source regression matched 70 arrays over 50 learning steps in the local audit. The notebook separately checks the frozen modulator. That does not reproduce the published paper's best benchmark accuracy.

## Why some old code remains

The frozen source retains an upstream `FIXME`, commented legacy lines and disabled compensation branches. The constructor rejects compensation, nonnegative-weight and loading options in this study. These branches are not selectable methods in this demo. I retained their exact bytes for protocol hashes and baseline regression rather than quietly clean up the historical implementation.

The new presentation helpers use readable names and separate statistics from plotting. No numerical training code was changed for this cleanup.
