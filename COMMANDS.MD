# Deterministic Experiment Engine Commands

This guide explains how to run the KuaiRand-Pure deterministic experiment engine manually. These commands run local Python and NumPy code. They do not contact Gemini or incur Vertex AI model charges.

## 1. Enter the project

Run all engine commands from the starter-kit directory:

```bash
cd kuairand-starter-kit
```

Relative paths such as `./KuaiRand-Pure/data` and `experiments/` assume this working directory.

## 2. Verify the installation

Run the complete test suite:

```bash
python3 -m unittest discover -s tests -v
```

Verify that protected baseline, data, evaluation, and published-score files have not changed:

```bash
python3 experiment_boundary.py --check
```

Inspect the current experiment-loop status:

```bash
python3 -m experiment_engine.controller status
```

The current workspace has one completed experiment (`E0001`). Status is
baseline-aware even though E0001's older registry record predates that logic:

```json
{
  "baseline_valid_primary": 0.6016,
  "best_candidate_decision": "reject_or_refine",
  "best_candidate_improvement": -0.0035498825073242424,
  "best_candidate_primary": 0.5980501174926758,
  "best_source": "stable_published_baseline",
  "best_valid_primary": 0.6016,
  "converged": false,
  "failed": 0,
  "iterations": 1,
  "remaining_iterations": 49,
  "successful": 1
}
```

## 3. Reproduce the published baseline

Run the organizer's pointwise FM with its published configuration and compare
its public validation metrics with `baseline_scores.json`:

```bash
python3 -m experiment_engine.reproduce_baseline
```

The command succeeds only if validation GAUC, nDCG@5, and primary score are
within `0.002` of the published values. It does not encode, score, or report the
test split. It can take about 40 seconds on a single CPU core.

To save the structured report:

```bash
python3 -m experiment_engine.reproduce_baseline \
  --output runs/baseline-reproduction.json
```

This reproduction command is separate from candidate selection. It does not consume an experiment ID or add a record to the experiment registry.

## 4. Run Phase 2 dataset diagnostics

Generate a deterministic profile using only train and validation data:

```bash
python3 -m experiment_engine.diagnostics profile \
  --output analysis/dataset-profile.json
```

The report includes row counts, cardinalities, label balance, duration
statistics, duplicate user-video rows, date/tab distributions, and train-to-
validation unseen-entity and label-rate drift. It is safe to provide to a future
Gemini orchestrator because it contains no test labels.

Inspect it with:

```bash
cat analysis/dataset-profile.json
```

Every candidate `result.json` also records deterministic validation subgroup
metrics and GAUC/nDCG/primary deltas against the protected published baseline.

## 5. Create and run a controlled experiment

Ask the controller to reserve the next unused experiment ID and create a
normalized specification from an approved template:

```bash
python3 -m experiment_engine.controller create \
  --template bpr_hybrid \
  --hypothesis "A BPR weight of 0.5 improves validation ranking."
```

The allocator scans templates, experiment packages, and the append-only registry.
Currently, E0001 is completed and E0002 is reserved inside the checked-in
ensemble example, so the next generated path is
`experiments/E0003/spec.json`. No E0002 model has been trained.

Open the generated file and change only the intended approved parameter:

```bash
nano experiments/E0003/spec.json
```

For this example, set:

```json
"bpr_weight": 0.5
```

Then run the exact path returned by `create`:

```bash
python3 -m experiment_engine.controller run experiments/E0003/spec.json
```

The controller will:

1. Validate the JSON specification and its parameters.
2. Verify protected-file hashes.
3. Reject duplicate experiment IDs.
4. Enforce epoch, wall-clock, and iteration budgets.
5. Load training and validation data while sealing the real test rows.
6. Train the approved model template.
7. Select the best checkpoint using validation primary score.
8. Save checkpoints and structured results.
9. Append one immutable summary to the experiment registry.
10. Compare the result with the protected published baseline or a better prior
    experiment.

Check the status after training:

```bash
python3 -m experiment_engine.controller status
```

Inspect the outputs:

```bash
cat experiments/E0003/spec.json
cat experiments/E0003/result.json
cat experiments/index.jsonl
ls -lh experiments/E0003/checkpoints
```

## 6. Create another experiment

Run the allocator again with one new hypothesis:

```bash
python3 -m experiment_engine.controller create \
  --template bpr_hybrid \
  --hypothesis "Eight negatives per positive improve validation ranking."
```

The output reports the reserved ID and path. Edit that file, change only the
parameter named by the hypothesis, and run it. Always use the returned path;
do not assume the next number or manually copy a template in a shared workspace.

For example, if the allocator returns `experiments/E0004/spec.json`:

```bash
nano experiments/E0004/spec.json
python3 -m experiment_engine.controller run experiments/E0004/spec.json
```

Each experiment should test one clear change. Each experiment ID can be used only once, including failed runs.

## 7. Run the ensemble template

Create an ensemble specification with an automatically allocated ID:

```bash
python3 -m experiment_engine.controller create \
  --template bpr_ensemble \
  --hypothesis "Three seeds and a popularity prior reduce candidate variance."
```

Edit and run the exact path returned by the command. For example, if it returns
`experiments/E0005/spec.json`:

```bash
nano experiments/E0005/spec.json
python3 -m experiment_engine.controller run experiments/E0005/spec.json
```

The ensemble trains three consecutive seeds and therefore takes longer than `bpr_hybrid`.

## 8. Experiment specification reference

A specification has this structure:

```json
{
  "schema_version": 1,
  "experiment_id": "E0003",
  "template": "bpr_hybrid",
  "data_dir": "./KuaiRand-Pure/data",
  "seed": 0,
  "parameters": {
    "embedding_dim": 16,
    "learning_rate": 0.001,
    "l2": 0.00001,
    "patience": 4,
    "negatives_per_positive": 4,
    "bpr_weight": 0.5,
    "bce_weight": 0.1,
    "popularity_weight": 0.0
  },
  "budget": {
    "max_epochs": 40,
    "max_wall_seconds": 21600
  },
  "hypothesis": "A BPR weight of 0.5 improves validation ranking."
}
```

Initially approved templates:

| Template | Behavior |
|---|---|
| `bpr_hybrid` | Trains one FM with within-user BPR and auxiliary BCE |
| `bpr_ensemble` | Trains three consecutive-seed BPR FMs and combines their scores |

Specifications cannot contain shell commands, Python source, module paths, or unknown parameters. The engine rejects anything outside the approved schema and template catalog.

## 9. Output locations

For a generated experiment `EXPERIMENT_ID`, the engine creates:

```text
experiments/EXPERIMENT_ID/
├── spec.json
├── result.json
└── checkpoints/
    ├── member-00.npz
    └── member-00.json

experiments/index.jsonl
```

Before execution, a reserved package contains only `spec.json`. A successful
run adds `result.json` and `checkpoints/`; a failed run adds `failure.json`.
The package directory therefore records the experiment's lifecycle without a
second standalone specification.

- `spec.json` is created first as the editable, reserved experiment
  specification and becomes the canonical specification when execution starts.
- `result.json` contains validation metrics, runtime, environment details, checkpoints, and the keep/reject comparison.
- `member-00.npz` contains model parameters.
- `member-00.json` contains checkpoint metadata and member validation metrics.
- `index.jsonl` is the append-only history of all successful and failed experiments.

An ensemble produces `member-00`, `member-01`, and `member-02` checkpoints.

## 10. Understanding the result

Important fields in `result.json` include:

```json
{
  "selection_split": "valid",
  "metrics": {
    "valid": {
      "GAUC": 0.0,
      "nDCG@5": 0.0,
      "primary": 0.0
    }
  },
  "comparison": {
    "reference": "stable_published_baseline",
    "previous_best": 0.6016,
    "candidate": 0.5981,
    "improvement": -0.0035,
    "epsilon": 0.002,
    "decision": "reject_or_refine"
  }
}
```

The primary metric is:

```text
primary = 0.5 * (GAUC + nDCG@5)
```

An experiment is marked `keep` only when it improves the protected published
validation baseline, or a better previous experiment, by more than `0.002`.
Otherwise, the decision is `reject_or_refine`.

The controller reports convergence after three consecutive successful experiments fail to improve the best validation primary by more than `0.002`.

## 11. Handling failures

If a run fails, replace `EXPERIMENT_ID` with the returned experiment ID and inspect:

```bash
cat experiments/EXPERIMENT_ID/failure.json
tail -n 1 experiments/index.jsonl
```

Fix the underlying problem, create a new specification with a new experiment ID, and run that new specification. Do not reuse or manually remove a registered ID; preserving failures is part of the audit trail.

Common checks are:

```bash
python3 experiment_boundary.py --check
python3 -m unittest discover -s tests -v
python3 -m experiment_engine.controller status
```

## 12. Human-gated final test and submission

Candidate runs never evaluate test data. After the team selects a genuine
finalist, a human must create an approval receipt using the exact confirmation
phrase:

```bash
python3 -m experiment_engine.approval FINALIST_ID \
  --approved-by "YOUR_NAME" \
  --confirm I_APPROVE_FINAL_TEST_AND_SUBMISSION
```

The approval is tied to the experiment's immutable specification fingerprint.
It cannot authorize a modified specification and cannot be granted twice.

Only after approval may the finalizer load test data, calculate final metrics,
and write a submission:

```bash
python3 -m experiment_engine.finalize FINALIST_ID
```

It writes:

```text
experiments/FINALIST_ID/final-result.json
runs/FINALIST_ID/submission.csv
```

Finalization is one-time: it refuses to overwrite an existing final result or
submission. Do not approve ordinary tuning runs.

## 13. What does and does not call Gemini

The following commands do not call Gemini:

```bash
python3 -m experiment_engine.reproduce_baseline
python3 -m experiment_engine.controller create --help
python3 -m experiment_engine.controller run experiments/EXPERIMENT_ID/spec.json
python3 -m experiment_engine.controller status
python3 -m experiment_engine.approval FINALIST_ID \
  --approved-by "YOUR_NAME" \
  --confirm I_APPROVE_FINAL_TEST_AND_SUBMISSION
python3 -m experiment_engine.finalize FINALIST_ID
python3 -m unittest discover -s tests -v
```

The only currently implemented command that contacts Gemini is the explicit live Vertex AI health check:

```bash
python3 -m agent.vertex_healthcheck --live
```

That health check does not choose or run experiments. The future Gemini orchestrator will inspect the registry, select an approved template, create a validated specification, invoke the same controller, and interpret the resulting evidence.

## Quick-start sequence

```bash
cd /Users/moob/Code/auto-ml/kuairand-starter-kit

python3 -m unittest discover -s tests -v
python3 experiment_boundary.py --check
python3 -m experiment_engine.controller status

python3 -m experiment_engine.reproduce_baseline

python3 -m experiment_engine.controller create \
  --template bpr_hybrid \
  --hypothesis "A BPR weight of 0.5 improves validation ranking."
# Read the experiment ID and path printed by the command.
# Edit that returned JSON so bpr_weight actually equals 0.5.
nano experiments/E0003/spec.json
python3 -m experiment_engine.controller run experiments/E0003/spec.json
python3 -m experiment_engine.controller status

cat experiments/E0003/result.json
```
