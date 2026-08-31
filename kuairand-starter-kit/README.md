# KuaiRand-Pure Starter Kit

## Requirements

Python 3.9+ and NumPy. **Nothing else.** PyTorch, pandas, and scikit-learn are not required.

## Data

Download the data from https://kuairand.com (direct Zenodo link; no registration required):

```bash
# Run this in the Starter Kit directory. The archive extracts to ./KuaiRand-Pure/.
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Baseline Boundaries

The organizer's pointwise FM is preserved in `official_baseline.py`, separately
from experimental candidates. Its interaction calculation uses equivalent,
non-mutating `np.einsum` dot products so the published implementation remains
reliable on Python 3.9-3.14. Its model, optimizer, fields, hyperparameters, and
early-stopping behavior are unchanged. Automated experiments must not modify
`official_baseline.py`, `data.py`, `evaluate.py`, or `baseline_scores.json`.
Verify their reviewed hashes with:

```bash
python3 experiment_boundary.py --check
```

The BPR models in `baseline.py` are experimental candidates, not substitutes
for official-baseline reproduction.

## Running the Models

Run the official baseline on train and public validation data:

```bash
python3 official_baseline.py
```

`--data_dir` defaults to `./KuaiRand-Pure/data`. Specify it explicitly if the data is stored elsewhere.

Run an experimental candidate separately:

```bash
python3 baseline.py --model bpr_ensemble
```

Available candidate `--model` values are `bpr_ensemble`, `fmbpr`, `pop`, and
`random`. `--data_dir` defaults to `./KuaiRand-Pure/data` for both commands.

## Reproduction and Controlled Experiments

Run the official pointwise FM with the published configuration and require its
public validation metrics to be within `0.002` of `baseline_scores.json`:

```bash
python3 -m experiment_engine.reproduce_baseline
```

This command uses the organizer's pointwise FM and unchanged training
configuration. The loader requests only training and validation, selects the
checkpoint using validation, and evaluates validation only. Test labels are not
read or materialized, and the report records `test_accessed: false`. The command
does not consume an experiment ID or participate in candidate selection. To save
its structured report:

```bash
python3 -m experiment_engine.reproduce_baseline --output runs/baseline-reproduction.json
```

Candidate experiments use strict, backward-compatible JSON specifications and
closed catalogues of approved model templates and pipeline operators. Ask the
controller to reserve the next unused experiment ID and create a default scalar
specification:

```bash
python3 -m experiment_engine.controller create \
  --template bpr_hybrid \
  --hypothesis "A BPR weight of 0.5 improves validation ranking."
# edit the approved scalar parameter in the returned path
python3 -m experiment_engine.controller run experiments/E0003/spec.json
python3 -m experiment_engine.controller status
```

The controller verifies protected hashes, enforces the iteration and wall-clock
budgets, prevents duplicate experiment IDs, and stops after convergence. The
runner removes the real test rows before candidate feature encoding and reports
validation metrics only. It packages the specification, atomic model checkpoints,
and structured run evidence together under `experiments/E####/`, then writes one
append-only registry record to `experiments/index.jsonl`.

### Phase 3 bounded research workflow

The Phase 3 helper reserves a one-variable-at-a-time BPR search from the closed
template catalogue. It skips parameter/seed combinations already represented by
an experiment specification:

```bash
python3 -m experiment_engine.phase3 plan --limit 8
# copy the returned paths into the run command after reviewing the hypotheses
python3 -m experiment_engine.phase3 run experiments/E####/spec.json
# after selecting a strong result, reserve seed replications
python3 -m experiment_engine.phase3 replicate experiments/E####/spec.json \
  --seeds 1 2
```

The deterministic policy layer can execute the next approved candidates without
manual per-experiment commands:

```bash
python3 -m experiment_engine.orchestrator --max-runs 3 --quiet
```

Add `--auto-continue` only when you explicitly want the orchestrator to open a
new, documented convergence window after the current one stops.

The Gemini integration is proposal-only: `agent.proposal` validates the model's
JSON against the approved templates and parameter bounds before the orchestrator
can create or run anything. A live model request is optional and may incur Vertex
AI charges; configuration-only health checks never call the model.

To ask Gemini for a proposal without training anything, save the validated
proposal as an immutable review artifact:

```bash
python3 -m agent.context --output runs/agent-context.json
python3 -m agent.run \
  --context runs/agent-context.json \
  --output-proposal runs/proposals/phase4-next.json
```

After reviewing the artifact, explicitly execute that exact saved proposal:

```bash
python3 -m agent.run \
  --proposal runs/proposals/phase4-next.json \
  --execute
```

The execution path does not call Gemini again. It revalidates the artifact,
checks its SHA-256 proposal fingerprint, and binds that fingerprint, the context
fingerprint, available Gemini token usage, source model, and one manual review
intervention into the immutable experiment specification. Modified proposals
are rejected instead of silently executing a different experiment.

For a bounded multi-step governed research run, use the agent loop. Omit
`--execute` to collect proposals only:

```bash
python3 -m agent.autonomous \
  --context analysis/dataset-profile.json \
  --max-steps 3 \
  --execute
```

Continuation after convergence remains human-gated unless `--auto-continue` is
explicitly supplied. Decisions are appended to `runs/agent-decisions.jsonl`.

The agent can build a fresh evidence context from the registry, specifications,
results, continuation history, and dataset diagnostics:

```bash
python3 -m agent.context --output runs/agent-context.json
python3 -m agent.autonomous --max-steps 3 --execute
```

Phase 3's strongest candidates were re-run with `--seed 1` and `--seed 2`; none
exceeded the protected baseline. All selection remains validation-only; test
evaluation still requires a separate human approval receipt. Phase 4 is
complete; Phase 5 advanced ranking is closed with a validation-only best
candidate, and Phase 6 autonomy work is the next milestone.

### Phase 4 governed operator workflow

Schema-version-2 specifications record a pipeline stage, one reviewed operator,
the evidence behind the hypothesis, and its expected effect. The first reviewed
operators are:

| Operator | Stage | Purpose |
|---|---|---|
| `missing_duration_category` | `cleaning` | Separate zero/missing duration from observed duration |
| `video_popularity_bucket` | `features` | Add a training-only video impression-count bucket |
| `inverse_duplicate_frequency` | `cleaning` | Inverse-frequency sample label-free exact training-feature duplicates |
| `smoothed_video_long_view_rate` | `features` | Bucket leave-one-out training video outcomes with a fixed smoothing prior |
| `user_activity_bucket` | `features` | Bucket training-only user interaction counts |
| `author_popularity_bucket` | `features` | Bucket training-only author impression counts |
| `user_tab_affinity` | `features` | Add a training-vocabulary user-tab interaction |
| `user_author_affinity` | `features` | Add a training-vocabulary user-author interaction |
| `video_tab_affinity` | `features` | Add a training-vocabulary video-tab interaction |
| `user_duration_affinity` | `features` | Add a user-duration-bucket interaction |
| `user_video_exposure_bucket` | `features` | Bucket causal prior user-video exposure counts |
| `video_recency_bucket` | `features` | Bucket days since the prior training appearance |
| `date_period_bucket` | `features` | Bucket dates using training-fitted boundaries |

Feature operators add the selected field without modifying protected `data.py`.
The duplicate operator instead supplies row-aligned training weights and leaves
the encoded train, validation, and test rows intact. Operator state and
vocabularies are fitted on training only; the same recorded operator is
reconstructed during human-approved finalization.

After reviewing Phase 3 convergence, explicitly open a new research window and
reserve one operator experiment:

```bash
python3 -m experiment_engine.controller continue \
  --reason "Begin Phase 4 reviewed cleaning and feature operators."
python3 -m experiment_engine.controller create \
  --template bpr_hybrid \
  --stage features \
  --operator video_popularity_bucket \
  --hypothesis "Training-only video popularity improves validation ranking." \
  --evidence "Validation diagnostics differ across item-popularity groups." \
  --expected-effect "Improve cold-item ordering without reducing GAUC."
```

The schema enforces one change per iteration: a cleaning or feature operator
cannot be combined with a model-parameter change. Existing schema-version-1
Phase 3 specifications retain their original serialization and fingerprints.

### Sandboxed candidate patches

Phase 4C candidate-code proposals are fingerprinted unified diffs restricted to
Python files under `candidates/`. Generate one without applying it:

```bash
python3 -m agent.context --output runs/patch-context.json
python3 -m agent.patch_run \
  --context runs/patch-context.json \
  --output-patch runs/patches/next.json
```

Validate it in a disposable clean checkout. This runs path and static policy,
compilation, imports, protected hashes, leakage/alignment tests, and the full
suite without changing the real worktree. When the artifact contains its
structured experiment payload, validation also mounts the dataset into the
checkout, trains through the deterministic runner, and records validation-only
result evidence:

```bash
python3 -m agent.candidate_patch runs/patches/next.json \
  --report runs/patch-reports/next.json
```

Promotion is a separate manual gate requiring the matching accepted report and
a sandbox result whose decision is `keep`:

```bash
python3 -m agent.candidate_patch runs/patches/next.json \
  --promote-report runs/patch-reports/next.json \
  --confirmation "PROMOTE REVIEWED CANDIDATE PATCH"
```

Generate the derived Phase 4 provenance, resource, recovery, and diff audit:

```bash
python3 -m agent.audit --output runs/phase4-audit-summary.json
```

Candidate decisions start from the protected published validation baseline,
not an empty registry. A result is kept only when it improves that baseline or a
better prior experiment by more than `0.002`.

Test evaluation and final submission require a separate human approval receipt:

```bash
python3 -m experiment_engine.approval E0005 \
  --approved-by "YOUR_NAME" \
  --confirm I_APPROVE_FINAL_TEST_AND_SUBMISSION
python3 -m experiment_engine.finalize E0005
```

Approval is bound to the experiment specification fingerprint. Finalization is
one-time and refuses to overwrite its test result or submission.

The initial approved templates are:

| Template | Purpose |
|---|---|
| `pointwise_fm` | Official-style pointwise FM for baseline-aligned feature controls |
| `pointwise_ensemble` | Three-seed pointwise FM ensemble for confirmed feature effects |
| `bpr_hybrid` | One FM trained with within-user BPR plus auxiliary BCE |
| `bpr_ensemble` | Three consecutive-seed hybrid-BPR FMs with an optional popularity prior |

Specifications cannot contain commands, Python paths, or source code. Unknown
fields, parameters, stages, and operators are rejected. Add reviewed model
templates to `experiment_engine/experiment_templates.py` and reviewed pipeline
operators to `candidates/feature_pipeline.py`.

## Implementation Status

The project follows the phases in the repository-level `ARCHITECTURE.md`:

| Phase | Status |
|---|---|
| Phase 0: official-baseline reproduction | Complete |
| Phase 1: deterministic experiment spine | Complete |
| Phase 2: EDA and subgroup diagnostics | Complete |
| Phase 3: first bounded research cycle | Complete (best candidate 0.60114; baseline 0.6016) |
| Phase 4: governed full-stack autonomy | Complete (reviewed operators, sandboxed patches, recovery, reflection, and audit evidence) |
| Phase 5: advanced ranking | Complete (E0046: 0.603005 validation primary; no test access) |
| Phase 6: task-completion campaign | Planned (single-agent research, sandboxed patches, sustained validation gains) |

Phase 6 uses a separate `campaigns/phase6/` workspace, preserving the completed
Phase 5 registry. Initialize it with `python3 -m experiment_engine.controller
--campaign phase6 init-campaign`. All campaign commands require the same
`--campaign phase6` flag. The agent records immutable live research-source
artifacts before proposing an experiment; fetched material is reference-only and
cannot supply executable instructions. Only sandbox-verified patches with
successful validation evidence can be auto-promoted. Final test access remains
human-gated and additionally requires a three-seed candidate at least `+0.002`
above the official validation primary.

Gemini proposals now use the schema-version-2 contract and can choose one
reviewed cleaning or feature operator, or one scalar change across loss, model,
and training stages. Fingerprinted sandboxed candidate patches are supported by
the governed Phase 4 workflow. See the repository-level `ARCHITECTURE.MD` for
its safety gates and completion criteria.

### Phase 5 closeout

Phase 5 first establishes a governed `pointwise_fm` candidate that reproduces
the protected official baseline while accepting reviewed feature operators.
Feature experiments name an explicit control experiment with the same template,
seed, parameters, budget, and data path. Results retain the global comparison
used for promotion and add a matched comparison used to decide whether an
individual component deserves replication.

The seed-zero screen evaluated `date_period_bucket`, `user_activity_bucket`, and
`user_video_exposure_bucket` against E0030. Date period and exposure advanced to
paired seeds one and two. Date period passed with positive effects in all three
seeds and a mean matched primary gain of `0.000755`; exposure was positive in
two seeds but missed the predeclared mean-gain threshold, and user activity was
rejected at seed zero. E0038 is the current best validation candidate at
`0.602369`.

No two-feature composite was justified by this screen. E0040–E0042 then
evaluated LambdaRank against the replicated pointwise-plus-date-period anchors.
All three matched primary deltas were positive, but their mean was only
`0.000110`, and mean matched `nDCG@5` gain was effectively zero (`0.000024`).
LambdaRank therefore does not advance. The three replicated date-period
checkpoints were then averaged in E0046, which reached `0.603005` primary
(`+0.001405` over the published baseline), with both GAUC and nDCG@5 improving
by approximately `0.00145`. It is the strongest validation candidate but still
remains below the conservative global promotion threshold. Phase 5 is closed:
the remaining experiment budget is deliberately retained for a separate Phase 6
autonomy campaign, rather than spent on unsupported follow-up variants. No test
rows were accessed and E0046 is not approved for finalization.

Create a matched feature experiment by passing the successful pointwise control
ID returned by the controller:

```bash
python3 -m experiment_engine.controller create \
  --template pointwise_fm \
  --stage features \
  --operator date_period_bucket \
  --control-experiment-id E0030 \
  --hypothesis "A training-fitted date period improves pointwise FM ranking." \
  --evidence "Validation positive-rate drift suggests a temporal effect." \
  --expected-effect "Improve primary score over the identical seed-zero control."
```

Run the deterministic Phase 2 dataset profile with:

```bash
python3 -m experiment_engine.diagnostics profile \
  --output analysis/dataset-profile.json
```

The profile uses only train and validation labels. Candidate experiment results
also contain validation subgroup metrics and metric deltas against the published
baseline.

## Future Gemini Orchestrator Setup

We recommend creating a separate environment with Python 3.12. The current system default,
Python 3.14, may be outside the version range verified by some agent SDKs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-agent.txt
```

Enable Vertex AI in a confirmed Google Cloud project and configure local Application Default
Credentials. These commands change your Google Cloud and local authentication state, so a project
member must run them manually:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com
gcloud auth application-default login
```

Set the runtime variables. Choose a model ID that is currently supported in the project's region and
supports function calling:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
export VERTEX_MODEL=YOUR_SUPPORTED_MODEL_ID
```

Validate the configuration only (no API call or model charges):

```bash
python -m agent.vertex_healthcheck
```

After authentication is complete, explicitly run one live check, which may incur charges:

```bash
python -m agent.vertex_healthcheck --live
```

## Task Definition (Fixed; Do Not Change)

| | |
|---|---|
| Task | **Within-user ranking** — for each user, rank only their impressions in the evaluation set; do not perform full-corpus retrieval |
| Relevance label | `long_view` (native binary column, 0/1) |
| Metrics | `GAUC` and `nDCG@5`; **the primary score is their mean** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Users with no positive examples | nDCG is set to 0.0 and included in the mean; GAUC includes only users where `0 < number of positives < number of impressions`, weighted by the number of positives |
| nDCG gain | `2^rel − 1` (equivalent to identity for binary labels) |

See `evaluate.py` for the implementation. Every convention is documented in the comments at the top
of that file.

## Baseline Ladder

Published hidden-test reference scores are shown below. They are documentation,
not output from the validation-only reproduction command. **The FM row is the
baseline to beat.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The True Metric Range: The nDCG@5 Ceiling Is 0.729, Not 1.0

Among the 23,875 users in the test set:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of the user's impressions are `long_view`) | **27.1%** | nDCG is always **0** and no model can improve it; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Users with distinguishable examples | **63.7%** | The effective sample for GAUC |

Therefore, even using the true labels as prediction scores—an oracle with perfect ranking—can achieve
only the following:

| | random | FM baseline | **oracle ceiling** | Share of available range captured by FM |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure evaluation progress against the oracle ceiling.** Treating 0.5946 as "far from a perfect
score of 1.0" is misleading: the baseline has already captured roughly 30% of the available range,
and the remaining headroom is 0.27 rather than 0.41.

Across five random seeds, FM has a standard deviation of **0.0008**. The convergence criterion is
therefore **ε = 0.002 (≈2.5σ), N = 3**: consider the process converged when the validation primary
score improves by no more than 0.002 for three consecutive iterations.

> Sanity check: if `--model random` does not produce a primary score of approximately 0.475 (±0.001),
> the evaluation harness is broken and should be fixed first.

## Submission Format

Submit a CSV with a header and one row for each row in the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Zero-based, consecutively increasing index corresponding to the row order of `data.load()[split]`. The order is deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`; after filtering by date, retain the original file order. |
| `user_id` / `video_id` | Redundant fields used only to validate alignment |
| `score` | The score assigned to the row by your model. It may be any real number; only relative order matters. NaN and Inf are not allowed. |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the evaluation set. In the
> test set, 3.06% of pairs are duplicated, with some appearing as many as 12 times. The pair therefore
> cannot serve as a primary key.

During development, generate and score only a validation example:

```bash
python3 submit.py validation-example.csv --make  --split valid
python3 submit.py validation-example.csv --check --split valid
python3 submit.py validation-example.csv --score --split valid
```

Do not generate or score a test submission during development. After convergence,
the approved final experiment uses `python3 -m experiment_engine.finalize E####`
to load test once, produce the final CSV, and record its result.

`--check` rejects an incorrect header, wrong row count, gaps in `row_id`, misaligned `user_id` or
`video_id` values, and scores that are non-numeric, NaN, or Inf. **Run `--check` yourself before
submitting.**

## Where to Start Making Changes

The ordering below is based on experiments, not guesses. Paths the organizers have already tested
and ruled out are clearly marked so you do not repeat the same work.

### Tested: These Two Approaches Did Not Help

| Approach tested | Result |
|---|---|
| **Adding static features** — using all 13 CWM feature fields (+`music_id`/`video_type`/`upload_type` + six coarse user buckets) | primary **0.5940** vs. **0.5950** with five fields: no difference beyond noise, and even a slight decrease |
| **Increasing model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887: virtually unchanged |

The reason is that the `user_id × video_id` interaction already captures most of the learnable signal.
Coarse buckets such as `follow_user_num_range` are redundant once `user_id` is present, while 1.14
million rows are not enough to support greater capacity. **Features and capacity are not the bottleneck.**

⚠️ Also note: **first-order terms for user-only features always contribute zero to the ranking.**
Because ranking is performed within each user, any term that is constant for a user cannot change the
within-group order. Experiments confirm that `item_pop × user bias` and plain `item_pop` produce scores
that are identical to the displayed precision. User features can help only through **interactions with
item-side features**.

### Unexplored: The Headroom Is Likely Here

Listed in our estimated order of promise (**the organizers have not tested these; they are left for you**):

1. **Change the loss function.** The current loss is pointwise log loss, but the metrics (GAUC and nDCG)
   are **ranking metrics**. Switch to a pairwise loss (BPR) or a listwise loss (softmax over a user's
   impressions) so that the objective matches the evaluation. We consider this the most promising option.
2. **User history sequences.** The existing features make **no use of behavioral sequences**. Each KuaiRand
   user has hundreds to thousands of training interactions, leaving DIN/SIM-style interest modeling
   entirely unexplored.
3. **Multiple objectives.** The logs also contain `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, and `play_time_ms`, which can serve as auxiliary tasks for the main `long_view` objective.
4. **Watch-time modeling.** This is precisely the contribution of [CWM](https://github.com/hyz20/CWM): it
   treats watch time as **censored regression**. Because the true watch time is truncated when a video
   ends, it uses a one-sided loss instead of squared error. This is a direction with substantial research depth.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since experiments indicate that capacity is not the
   bottleneck, prioritize items 1–4 first.
6. **Time features and distribution shift.** Explore `hourmin`, `date`, and the shift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` contains 1.18 million randomly
   exposed examples. It can serve as an additional unbiased validation set to detect whether a model is
   overfitting to biased traffic.

## Using Your Own Model (Including CWM)

`evaluate.py` is fully decoupled from the model. It requires only three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores may come from any model
```

- `user_ids`: the `user_id` for each row in the evaluation set
- `labels`: the row's `long_view` value (0/1)
- `scores`: the score assigned to the row by your model (any real number; only relative order matters)

You can therefore skip `baseline.py` entirely and use PyTorch, LightGBM, or xDeepFM from
[CWM](https://github.com/hyz20/CWM), as long as you pass the resulting `scores` to `evaluate()`.
**`evaluate.py` is the sole authority on scoring conventions.**

> A note about CWM: it depends on `torch==1.6.0` (a 2020 release that is unlikely to install on recent
> GPUs), optimizes counterfactual watch time, and evaluates against its own reconstructed `long_view2`
> label. It is research code for a paper on watch-time debiasing. Treat it as an **advanced reference**,
> not a recommended starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementations and all scoring conventions. **Do not modify.** |
| `data.py` | Protected canonical data loading, splits, and five-field encoding. |
| `official_baseline.py` | Protected organizer pointwise FM with the Python 3.14-safe logits calculation. |
| `baseline.py` | Experimental BPR candidates plus popularity and random references. |
| `baseline_scores.json` | Official published scores, seed variance, and convergence parameters. |
| `submit.py` | Generates and validates submission files. |
| `ablation_features.py` | Feature ablation experiments that reproduce the finding that additional features do not help. |
| `experiment_engine/` | Deterministic specifications, controller, runner, checkpoints, registry, approval, and finalization. |
