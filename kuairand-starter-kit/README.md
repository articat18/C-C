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

The organizers' original pointwise FM is frozen in `official_baseline.py`. It is kept separate from
the experimental candidate models. Automated agents must not modify `official_baseline.py`, `data.py`,
`evaluate.py`, or `baseline_scores.json`. Use the following command to verify the content hashes of
these files:

```bash
python3 experiment_boundary.py --check
```

The enhanced model that currently achieves the published score is a three-seed BPR ensemble; it
should not be described as the "unmodified official FM."

## Running the Models

```bash
python3 baseline.py --model bpr_ensemble
```

`--data_dir` defaults to `./KuaiRand-Pure/data`. Specify it explicitly if the data is stored elsewhere.

Available `--model` values are `bpr_ensemble` (enhanced candidate), `fmbpr` (single-model experiment),
`pop` (trivial baseline), and `random` (lower bound, used to sanity-check the evaluation code).

Run the organizers' original implementation separately:

```bash
python3 official_baseline.py
```

## Reproduction and Controlled Experiments

Run the stable three-seed BPR FM candidate with the published configuration and
require every validation/test metric to be within `0.002` of
`baseline_scores.json`:

```bash
python3 -m experiment_engine.reproduce_baseline
```

This is an explicit baseline-reproduction command, not part of candidate model
selection. The frozen pointwise implementation remains in `official_baseline.py`
for provenance and semantics tests, but it is known to train unstably; the BPR
ensemble is the score-reproducing implementation. The reproduction command
trains on the full local dataset and can take several minutes. To also save its
structured report under an agent-editable directory:

```bash
python3 -m experiment_engine.reproduce_baseline --output runs/baseline-reproduction.json
```

Candidate experiments use strict JSON specifications and a closed catalog of
approved templates. Start by copying a template and assigning a new experiment
ID; never run the checked-in example ID more than once:

```bash
cp experiments/templates/bpr_hybrid.json experiments/E0003.json
# edit experiment_id, hypothesis, and approved scalar parameters
python3 -m experiment_engine.controller run experiments/E0003.json
python3 -m experiment_engine.controller status
```

The controller verifies protected hashes, enforces the iteration and wall-clock
budgets, prevents duplicate experiment IDs, and stops after convergence. The
runner removes the real test rows before candidate feature encoding and reports
validation metrics only. It writes atomic model checkpoints under `checkpoints/`,
structured run evidence under `experiments/E####/`, and one append-only registry
record to `experiments/index.jsonl`.

The initial approved templates are:

| Template | Purpose |
|---|---|
| `bpr_hybrid` | One FM trained with within-user BPR plus auxiliary BCE |
| `bpr_ensemble` | Three consecutive-seed hybrid-BPR FMs with an optional popularity prior |

Specifications cannot contain commands, Python paths, or source code. Unknown
fields and parameters are rejected. Add a reviewed template to
`experiment_engine/experiment_templates.py` when a genuinely new experiment
family is ready.

## Vertex AI API (Phase 2)

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

Scores on the test set. **The FM row is the baseline to beat.**

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

Generate and validate a submission:

```bash
python3 submit.py --make  --split test  submission.csv    # Generate a sample submission with the official FM baseline
python3 submit.py --check --split test  submission.csv    # Validate its format and alignment
python3 submit.py --score --split valid submission.csv    # Validate and score it (local validation data only)
```

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
| `data.py` | Data loading, official splits, and feature encoding. Modify this file when adding features. |
| `baseline.py` | Three baselines. FM is the one to beat. |
| `baseline_scores.json` | Official published scores, seed variance, and convergence parameters. |
| `submit.py` | Generates and validates submission files. |
| `ablation_features.py` | Feature ablation experiments that reproduce the finding that additional features do not help. |
