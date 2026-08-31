# Phase 8 final results

## Selected final model

The final candidate is **E0016**, a three-seed `sequence_ensemble` using an
embedding dimension of 16 and hidden dimension of 16. It was selected over the
slightly higher single-seed E0008 result because E0016 supplies the required
replication evidence. Final approval was granted before the one final test
evaluation.

| Split | GAUC | nDCG@5 | Primary |
| --- | ---: | ---: | ---: |
| Official FM baseline, validation | 0.667400 | 0.535700 | 0.601600 |
| E0016, validation | 0.670338 | 0.538003 | 0.604170 |
| Validation delta | +0.002938 | +0.002303 | +0.002570 |
| Official FM baseline, test | 0.661000 | 0.528200 | 0.594600 |
| E0016, final test | 0.664927 | 0.530630 | 0.597778 |
| Test delta | +0.003927 | +0.002430 | +0.003178 |

For transparency, E0008 was the campaign's highest individual validation run
at 0.604581, but it was not selected because it had one member rather than the
three-seed evidence required for campaign finalization.

## Final artifact bundle

This directory contains the final submission, E0016 specification, validation
result, final-test result, human approval receipt, and all three member
checkpoints with metadata. The source campaign remains in the ignored
`campaigns/phase8/` workspace.

| File | Purpose |
| --- | --- |
| `E0016-submission.csv` | Final KuaiRand-Pure submission |
| `spec.json` | Immutable final experiment specification |
| `result.json` | Validation metrics and member evidence |
| `final-result.json` | Final test metrics and submission provenance |
| `approval.json` | Human approval receipt for test access |
| `checkpoints/member-*.npz` | Three final model checkpoint members |
| `checkpoints/member-*.json` | Checkpoint metadata |

## Iteration log

All model selection used the training and validation splits only. No test data
was read until finalizing E0016.

| ID | Candidate | Status | Validation primary | Hypothesis / outcome |
| --- | --- | --- | ---: | --- |
| E0001 | Pointwise FM control | Success | 0.601470 | Establish a matched control. |
| E0002 | Causal attention | Recovered failure | — | Empty sealed test split exposed an encoder shape edge case. |
| E0003 | Causal attention | Recovered failure | — | History vocabulary-offset assumption was invalid. |
| E0004 | Causal attention, h=32 | Success | 0.603249 | Attend over eight strictly-prior positive videos. |
| E0005 | Causal attention, h=32 | Success | 0.603249 | Auditable recovery replication. |
| E0006 | Causal attention, h=64 | Reserved only | — | Interrupted reservation; no execution result. |
| E0007 | Causal attention, h=64 | Success | 0.602940 | Test a wider scoring head; rejected. |
| E0008 | Causal attention, h=16 | Success | 0.604581 | Test lower capacity for stronger regularization; raw validation best. |
| E0009 | Causal attention, emb=32 | Success | 0.603056 | Test greater embedding capacity; rejected. |
| E0010 | Pointwise FM + date period | Success | 0.602159 | Screen a training-fitted temporal feature. |
| E0011 | Pointwise FM + exposure | Success | 0.602021 | Screen prior user-video exposure. |
| E0012 | Pointwise FM + popularity | Success | 0.601893 | Screen training-only item popularity. |
| E0013 | Causal attention, L2=1e-5 | Success | 0.602936 | Test stronger regularization; rejected. |
| E0014 | Causal attention, batch=4096 | Success | 0.603942 | Test smaller-batch updates; rejected. |
| E0015 | Causal attention, batch=2048 | Success | 0.602728 | Test additional gradient noise; rejected. |
| E0016 | Three-seed sequence ensemble | Success / finalized | 0.604170 | Replicate a compact sequence model and finalize it. |

## Code changes and recovery evidence

The competitive sequence direction was added in these commits:

| Commit | Change |
| --- | --- |
| `3b78705` | Added causal attention history encoding, ranker, template, runner, checkpoint, finalization, and control support. |
| `5b95919` | Fixed empty sealed-split attention encoding and added regression coverage. |
| `50f5a1a` | Preserved the true categorical history offset and added regression coverage. |

The two execution failures were preserved as E0002 and E0003 rather than
overwritten. Their fixes were committed and tested before retrying E0004. The
agent also recorded controller-contract corrections when a proposed matched
control had immutable parameter mismatches.

## Autonomy and resources

| Measure | Value |
| --- | ---: |
| Recorded iterations | 15 of 50 |
| Successful experiments | 13 |
| Recovered failures | 2 |
| Manual interventions in agent provenance | 0 |
| Agent decision wall-clock time | 926.935 seconds |
| Training time across successful experiments | 631.428 seconds |
| Gemini input tokens | 70,382 |
| Gemini output tokens | 2,882 |
| Gemini thought tokens | 31,790 |
| Gemini total tokens | 105,054 |
| GPU hours | 0.0 (CPU NumPy) |

The source evidence was the saved SASRec reference, `S-f547f2b843481b32`.
