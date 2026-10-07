# Review before publication

Ning Yang authorized GitHub publication on 7 October 2026 after local review and validation. Publication preserves the existing packaging history and original commit timestamps. This authorization does not include sending email to the supervisor.

## What the supervisor should see first

1. README: the CSDP limitation, the one-eighth dependency change and the conditional accuracy result.
2. `demo.ipynb`: the real-source check, all seed results, the worst-seed curve, budget sensitivity and one fresh paired run.
3. `docs/appendix.pdf`: the two-page proposal appendix with the same numerical conclusions.

The README and notebook must identify JAX/ngclearn as the actual framework. snnTorch was an example in the supervisor's email; it is not a dependency of this implementation.

## Local acceptance checks

- Verify frozen source and protocol hashes, full seed coverage and paired random-stream records.
- Restart the notebook kernel and run all cells. Keep consecutive execution counts and short outputs.
- Check the original, new five-epoch and new twenty-epoch intervals separately. Do not substitute the best group.
- Keep the four losses greater than one percentage point and the -3.90 pp worst seed visible.
- Distinguish SD shading from confidence intervals. Do not claim a cause for the worst-seed reversal.
- Scan tracked text and decoded notebook outputs for non-English prose, workstation paths and credentials.
- Exclude environments, logs, large arrays, checkpoints, ZIPs and third-party paper PDFs.
- Review the PDF and notebook together. Installation in a fresh environment and other platforms must not be described as tested unless actually tested.

## After explicit approval

Publish the reviewed local commit history without inventing dates or implying these packaging commits were the original experiment history. Verify the repository's notebook rendering and the exact commit to be shared.

Only then create the annotated tag `v1.0-proposal-appendix` and its GitHub Release. Suggested release text:

> This tagged version is the frozen appendix for my PhD project proposal. The tag will not be moved. Subsequent exploratory work will be placed on separate branches.

Keep `main` at that reviewed version. Use `mechanism-study` for new hypotheses and experiments. If an error is discovered, disclose it and issue a separately identified correction; do not silently replace the tagged evidence.

Send the supervisor the repository and release links with the appendix PDF after approval. The accompanying note should state the measured structural improvement and the accuracy limitation, then ask whether this is a useful direction to pursue. It should not claim a novel algorithm or a universal improvement.
