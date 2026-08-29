# Deterministic Experiment Engine Commands

This guide explains how to run the KuaiRand-Pure deterministic experiment engine manually. These commands run local Python and NumPy code. They do not contact Gemini or incur Vertex AI model charges.

## 1. Enter the project

Run all engine commands from the starter-kit directory:

```bash
cd kuairand-starter-kit
```

Relative paths such as `./KuaiRand-Pure/data`, `experiments/`, and `checkpoints/` assume this working directory.

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

A new registry reports zero iterations:

```json
{
  "best_valid_primary": null,
  "converged": false,
  "failed": 0,
  "iterations": 0,
  "remaining_iterations": 50,
  "successful": 0
}
```

## 3. Reproduce the published baseline

Run the stable three-seed BPR ensemble and compare its validation and test metrics with `baseline_scores.json`:

```bash
python3 -m experiment_engine.reproduce_baseline
```

The command succeeds only if GAUC, nDCG@5, and primary score on both splits are within `0.002` of the published values. It can take several minutes.

To save the structured report:

```bash
python3 -m experiment_engine.reproduce_baseline \
  --output runs/baseline-reproduction.json
```

This reproduction command is separate from candidate selection. It does not consume an experiment ID or add a record to the experiment registry.

## 4. Run the first controlled experiment

The included hybrid-BPR template uses experiment ID `E0001`:

```bash
python3 -m experiment_engine.controller run \
  experiments/templates/bpr_hybrid.json
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
10. Compare the result with the best previous experiment.

Check the status after training:

```bash
python3 -m experiment_engine.controller status
```

Inspect the outputs:

```bash
cat experiments/E0001/spec.json
cat experiments/E0001/result.json
cat experiments/index.jsonl
ls -lh checkpoints/E0001
```

## 5. Create another experiment

Copy an approved template to a new specification file:

```bash
cp experiments/templates/bpr_hybrid.json experiments/E0003.json
```

Open it in an editor:

```bash
nano experiments/E0003.json
```

At minimum, assign the matching unused experiment ID and describe one hypothesis:

```json
{
  "experiment_id": "E0003",
  "hypothesis": "A BPR weight of 0.5 improves validation ranking."
}
```

Change only the intended parameter. For this example:

```json
"bpr_weight": 0.5
```

Keep the rest of the copied JSON document intact, save it, and run:

```bash
python3 -m experiment_engine.controller run experiments/E0003.json
python3 -m experiment_engine.controller status
```

Each experiment should test one clear change. Each experiment ID can be used only once, including failed runs.

## 6. Run the ensemble template

Copy the ensemble template and assign a new ID:

```bash
cp experiments/templates/bpr_ensemble.json experiments/E0004.json
nano experiments/E0004.json
```

Ensure the JSON contains:

```json
"experiment_id": "E0004"
```

Then run it:

```bash
python3 -m experiment_engine.controller run experiments/E0004.json
```

The ensemble trains three consecutive seeds and therefore takes longer than `bpr_hybrid`.

## 7. Experiment specification reference

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

## 8. Output locations

For experiment `E0003`, the engine creates:

```text
experiments/E0003/
├── spec.json
└── result.json

checkpoints/E0003/
├── member-00.npz
└── member-00.json

experiments/index.jsonl
```

- `spec.json` is the normalized specification that was executed.
- `result.json` contains validation metrics, runtime, environment details, checkpoints, and the keep/reject comparison.
- `member-00.npz` contains model parameters.
- `member-00.json` contains checkpoint metadata and member validation metrics.
- `index.jsonl` is the append-only history of all successful and failed experiments.

An ensemble produces `member-00`, `member-01`, and `member-02` checkpoints.

## 9. Understanding the result

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
    "previous_best": null,
    "improvement": null,
    "epsilon": 0.002,
    "decision": "keep"
  }
}
```

The primary metric is:

```text
primary = 0.5 * (GAUC + nDCG@5)
```

An experiment is marked `keep` when it is the first successful experiment or improves the previous best validation primary by more than `0.002`. Otherwise, the decision is `reject_or_refine`.

The controller reports convergence after three consecutive successful experiments fail to improve the best validation primary by more than `0.002`.

## 10. Handling failures

If a run fails, inspect:

```bash
cat experiments/E0003/failure.json
tail -n 1 experiments/index.jsonl
```

Fix the underlying problem, create a new specification with a new experiment ID, and run that new specification. Do not reuse or manually remove a registered ID; preserving failures is part of the audit trail.

Common checks are:

```bash
python3 experiment_boundary.py --check
python3 -m unittest discover -s tests -v
python3 -m experiment_engine.controller status
```

## 11. What does and does not call Gemini

The following commands do not call Gemini:

```bash
python3 -m experiment_engine.reproduce_baseline
python3 -m experiment_engine.controller run experiments/E0003.json
python3 -m experiment_engine.controller status
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

python3 -m experiment_engine.controller run experiments/templates/bpr_hybrid.json
python3 -m experiment_engine.controller status

cat experiments/E0001/result.json
```
