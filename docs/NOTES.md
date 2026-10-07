# A route I stopped

Before grouping, I tried changing the weight-decay treatment under fixed neuron-sign constraints. In three development seeds, the magnitude-decay variant gained 2.228 pp, tied, and lost 1.114 pp against native decay: +0.371 pp on average. It missed the continuation rule, which required at least +1 pp on average and a positive difference in every seed against both comparators. I stopped that candidate.

This was a different constraint, not evidence that CSDP's original update is wrong. I did not carry the failed candidate into the grouped comparison. The [development summary and continuation rule](abandoned_trial.json) retain the nine outcomes. Full source and run logs remain in the local archive; this notebook focuses on grouping.

# What remains unexplained

The worst grouped seed is not a clean story of divergence: its paired differences over epochs 1–5 are -3.62, +5.57, -2.79, -2.51 and -3.90 pp. Epoch 3 reverses the advantage seen at epoch 2. This observation does not identify a cause. Missing cross-group information and changed objective scaling are hypotheses for later work.
