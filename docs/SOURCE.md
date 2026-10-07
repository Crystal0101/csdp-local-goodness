# Source and scope

The two `studies/*/source/` trees contain the frozen CSDP implementation used in the experiments, with the upstream LICENSE preserved. `source_manifest.json` gives the SHA-256 of every source file. The frozen runners are kept byte-for-byte because each protocol records its runner hash. The portable wrapper supplies paths; it does not change their update equations.

The adaptation is in `custom/goodnessModCell.py`: fixed equal contiguous groups, scaled logits, and a mean group BCE. G=1 retains the original branch. The supplement changes only the allowed group-count guard and its error message; see its `source_change_record.json`.

Historical protocols may mention original local paths. They are records, not the runnable entry point; use `run.py` with the paths documented here.

The upstream author-source regression matched 70 arrays over 50 learning steps in the local audit. The notebook separately checks the frozen modulator. That does not reproduce the published paper's best benchmark accuracy.
