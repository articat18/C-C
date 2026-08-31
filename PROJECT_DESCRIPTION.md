# Project Description

## Problem statement

TikTok TechJam 2026 Track 2 ("Autonomous ML Research Agent for Recommender Systems") asks teams to build an agent that autonomously runs the standard MLE iteration loop — read problem → inspect data (EDA) → engineer features → train/tune → evaluate → reflect/revise — on the KuaiRand-Pure benchmark, without a human writing the per-iteration code.

Fixed by the organizers:

- **Task**: within-user ranking (not full-catalog retrieval) of each user's logged impressions.
- **Label**: `long_view` (native binary column).
- **Metrics**: `GAUC` (per-user AUC, weighted by positive count, users with 0% or 100% positives excluded) and `nDCG@5` (gain `2^rel − 1`; all-negative users score 0 and are included). Primary score = `mean(GAUC, nDCG@5)`.
- **Splits**: date-based — train `20220408–20220421` (1,141,112 rows), valid `20220422–20220428` (124,909 rows), hidden test `20220429–20220508` (170,588 rows).
- **Official baseline**: organizer-provided pointwise FM (`k=16, lr=1e-3`, 5 categorical fields), numpy-only, ~40s/CPU-core. Published validation: GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016. Published hidden-test (5-seed mean, σ=0.0008): GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946.
- **Convergence rule**: ε = 0.002, N = 3 (no validation-primary improvement > ε over 3 consecutive iterations), hard cap 50 iterations / 6h wall-clock.
- **Constraint**: no external training data or pretrained weights fit on these benchmarks' test labels; agent develops on train+valid only, hidden test evaluated once on the final designated submission.

## How our solution addresses it

An LLM given unrestricted write access to a scoring/evaluation pipeline is a classic self-sabotage/reward-hacking risk: it can mutate `evaluate.py`, select checkpoints using leaked test signal, silently overwrite prior results, or exceed iteration/wall-clock budgets to farm more attempts. Because the challenge explicitly scores Verification & Robustness alongside raw score improvement, we split the system into a deterministic, hash-guarded execution layer and an LLM policy layer that is only ever allowed to *propose*:

```
Gemini proposal layer (judgment only, agent/proposal.py + agent/context.py)
  inspect evidence → choose stage → propose one bounded change
        |                                   ^
   validated proposal                   audit evidence
        v                                   |
Deterministic orchestrator + controller (no LLM in the loop)
  boundary/leakage/budget checks → sandboxed execution → checkpoint select → registry
```

Gemini never trains a model, computes a metric, or picks a checkpoint. It selects a pipeline stage (`cleaning`, `features`, `loss`, `model`, or `training`), a reviewed operator or one bounded source patch, and states its hypothesis; everything after that — validation, leakage checks, training, evaluation, checkpointing, and the keep/reject/refine decision's evidence — runs through code the model cannot alter.

**Foundation (Phases 0–2).** `official_baseline.py` reproduces the organizer's pointwise FM with non-mutating dot-product logits; `experiment_engine/reproduce_baseline.py` re-derives it from `data.py` on train+valid only and confirms it lands within `0.002` of `baseline_scores.json`. `experiment_boundary.py` SHA-256-locks the four protected files (`data.py`, `evaluate.py`, `official_baseline.py`, `baseline_scores.json`), restricts writes to an explicit allowlist (`agent/`, `candidates/`, `experiment_engine/`, `experiments/`, `analysis/`, `runs/`), and hard-fails any checkpoint selection against anything but the `valid` split. `experiment_engine/diagnostics.py` produces a deterministic dataset profile and per-subgroup metrics (duration bucket, item popularity, user activity, date period, label composition) that seed later hypotheses.

**Governed autonomy (Phases 3–4, complete on `main`).** `experiment_engine/orchestrator.py` is a deterministic policy layer — it owns no training code, only decides what to run next and interprets results. `agent/proposal.py` constrains Gemini to one reviewed operator (from `candidates/feature_pipeline.py` — e.g. `missing_duration_category`, `video_popularity_bucket`, `inverse_duplicate_frequency`, `smoothed_video_long_view_rate`, user/author/tab affinity buckets) or one bounded source patch confined to `candidates/`, validated and fingerprinted before execution so the same proposal can be re-run without a second model call. `agent/candidate_patch.py` applies and tests any proposed source patch in an isolated clean checkout, gated by path validation, static policy checks, leakage/row-alignment tests, protected-file hashes, and the full test suite, before it is ever allowed to train. `agent/audit.py` derives a complete audit summary (diffs, token usage, wall-clock, GPU-hours, failures, recovery events, manual interventions) from the append-only registry.

Recovery is exercised, not just designed: **E0023** (inverse-frequency duplicate weighting) failed outright, and **E0024** — the same hypothesis, repaired — succeeded, logged as a failure/recovery pair in `experiments/index.jsonl`. **E0026** ran as a zero-manual-intervention autonomous step that reflected on subgroup evidence and proposed the next change itself.

**Current results.** 28 experiments are logged (`experiments/index.jsonl`): 27 succeeded, 1 failed-then-recovered. The best candidate to date is **E0011 at validation primary 0.60114** — still fractionally under the protected baseline's 0.6016, so per the promotion rule (must beat the baseline by ≥ 0.002, survive 3 seeds) nothing has been promoted yet. Every one of the 28 is a `reject_or_refine` decision the system reached on its own evidence, which is itself the point: the guarded loop correctly declines to promote a candidate that doesn't clear the bar, rather than reporting a misleadingly good number.

Next: Phase 5 (advanced ranking objectives — LambdaRank, historical/sequence features — gated behind the Phase 4 diagnostic evidence actually justifying them) and optional Phase 6 multi-agent decomposition, with the orchestrator remaining sole owner of experiment state, budgets, convergence, and final-submission approval throughout.

## Development tools

- **VS Code** — primary editor.
- **Claude Code** (Anthropic) — AI coding assistant used throughout development, including diagnosing and fixing a Windows-checkout CRLF issue that broke the SHA-256 boundary guard, and reverting an unreviewed protected-file mutation from a teammate commit on a parallel branch.
- **Codex** (OpenAI) — a second AI coding agent, directed against `ARCHITECTURE.MD` as its implementation spec for building out the experiment engine and governed-autonomy phases.
- **Git / GitHub** — version control (`github.com/articat18/C-C`).
- Python 3 CLI workflow, cross-platform (developed on macOS, also verified on Windows).

## APIs used

- **Google Gemini via Vertex AI**, through the `google-genai` SDK (`agent/proposal.py`, `agent/candidate_patch.py`). Authenticated via Application Default Credentials (`agent/config.py`, validated by `agent/vertex_healthcheck.py`), not API keys. This is the model actually driving the Phase 3–4 autonomous proposals recorded in the experiment registry — it selects the pipeline stage and change to try, but has no code path to training, evaluation, checkpoint selection, or the hidden test set.
- No other third-party APIs. All training/evaluation is local and offline against the on-disk KuaiRand-Pure CSVs.

## Libraries and frameworks

- **NumPy** — the sole numerical dependency for the model itself (FM embeddings, dot-product logits, BCE + within-user BPR loss, SGD-style training loop). No PyTorch/TensorFlow/scikit-learn/pandas, per the starter kit's constraint and to keep every run auditable and fast (~40s, single CPU core, no GPU).
- **Python standard library** — `csv`/`collections` for data loading (`data.py`), `hashlib` for the boundary and proposal-fingerprint checks, `json`/`pathlib` for experiment specs, the registry, and checkpoints, `ast` for static policy checks on candidate patches, `argparse` for all CLI entry points, `unittest` for the automated test suite covering config, controller, orchestrator, data, diagnostics, boundary/spec/runner/storage, finalization, and official-baseline regression.
- **`google-genai` / `google-adk`** (Google Gen AI SDK, Agent Development Kit) + **`python-dotenv`** — the Gemini proposal client and its environment/config validation.

## Datasets and assets

- **KuaiRand-Pure** (Kuaishou, released via Zenodo, no-registration public research release) — ~1.4M interaction rows, ~27K users × ~7.6K videos, 12 feedback signals per impression (click/like/follow/comment/forward/`long_view`/`play_time_ms`/…), plus a separate randomized-exposure log (`log_random_4_22_to_5_08_pure.csv`, 1.18M rows) usable for unbiased off-policy diagnostics. Loaded via `data.py`, which reads `log_standard_4_08_to_4_21_pure.csv` + `log_standard_4_22_to_5_08_pure.csv` and `video_features_basic_pure.csv`, date-filters into the fixed splits, and preserves file/row order for deterministic `row_id` alignment.
- **`baseline_scores.json`** — organizer-published FM baseline scores, 5-seed variance, and convergence parameters; one of the four SHA-256-protected reference files.
- No external, scraped, or manually labelled data — the challenge's sole hard constraint (no external training data or pretrained weights fit on the benchmark's test labels) is enforced by keeping `data.py` itself hash-protected, and by confining every agent-authored feature/patch to `candidates/`.

## Current status

Phases 0–4 of the six-phase architecture are complete on `main`: baseline reproduction, the guarded deterministic experiment spine, dataset diagnostics, the first bounded structured search, and governed full autonomy (typed Gemini proposals → reviewed operators or sandboxed patches → deterministic execution → reflection/recovery → audit trail). 28 experiments are on record, with the strongest candidate (E0011, primary 0.60114) not yet clearing the baseline by the required margin. Phase 5 (advanced ranking objectives, pursued only where Phase 4's diagnostic evidence justifies the added complexity) is next.
