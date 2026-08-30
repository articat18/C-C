# AutoML Agent Architecture

This architecture is based on the design material under `assets/` and the current KuaiRand-Pure starter kit. It deliberately starts with one judgment-bearing orchestrator agent surrounded by deterministic experiment infrastructure. Multi-agent coordination is deferred until the single-agent experiment loop is reliable and reproducible.

## Architecture Principles

1. Only work requiring judgment belongs to the agent.
2. Training, evaluation, checkpoint selection, logging, budget enforcement, and submission validation remain deterministic programs.
3. The validation split is the only model-selection split.
4. The test split stays sealed during experimentation and is used only for the final selected model.
5. Every iteration tests one clear hypothesis.
6. All transformations and historical features are fitted using training data and past events only.
7. Every experiment must be reproducible from its recorded specification.
8. The orchestrator remains the sole owner of experiment state, budgets, convergence decisions, and final submission approval.

## System Overview

```text
                         Human approval
                              |
                    final test / submission
                              |
+--------------------------------------------------------------+
|                    Orchestrator agent                        |
| inspect -> hypothesize -> choose one change -> reflect       |
+---------------+-------------------------------^--------------+
                | experiment specification      | evidence
                v                               |
+--------------------------------------------------------------+
|              Deterministic experiment controller             |
| boundary check | budget | runner | checkpoint | result logger|
+-------+---------------+---------------+---------------+------+
        |               |               |               |
        v               v               v               v
 Data/feature       Candidate model   Fixed evaluator   Run registry
 pipeline           and loss          evaluate.py       experiments/
        |               |               |
        +---------------+---------------+
                        |
             train -> validate -> diagnose
```

The agent decides what to try and interprets the evidence. It does not calculate metrics, choose checkpoints through intuition, mutate evaluation code, or manually manage experiment state.

## Components

| Component | Responsibility | Proposed location |
|---|---|---|
| Orchestrator | Inspect results, form one hypothesis, choose an experiment, and decide whether to keep, reject, or refine it | `kuairand-starter-kit/agent/orchestrator.py` |
| Experiment specification | Immutable description of features, model, loss, hyperparameters, seeds, and budgets | `kuairand-starter-kit/experiments/E####/spec.json` |
| Controller | Enforce boundaries, validation-only selection, iteration limits, wall-clock limits, and convergence | `kuairand-starter-kit/experiment_engine/controller.py` |
| Experiment runner | Load a specification, train deterministically, restore the best checkpoint, and return structured evidence | `kuairand-starter-kit/experiment_engine/experiment_runner.py` |
| Feature pipelines | Leakage-safe feature transformations fitted only on training history | `kuairand-starter-kit/candidates/features/` |
| Models | FM/BPR initially; later ranking, sequence, and multitask candidates | `kuairand-starter-kit/candidates/models/` |
| Losses and samplers | BCE, within-user BPR, LambdaRank, and negative-sampling strategies | `kuairand-starter-kit/candidates/losses/` |
| Diagnostics | Aggregate and subgroup GAUC/nDCG analysis | `kuairand-starter-kit/experiment_engine/diagnostics.py` |
| Run registry | Append-only specifications, metrics, timings, environment, diffs, decisions, and failures | `kuairand-starter-kit/experiments/index.jsonl` |
| Checkpoints | Best validation checkpoint and minimal run artifacts | `kuairand-starter-kit/experiments/E####/checkpoints/` |
| Submission gate | Generate and validate an aligned CSV after explicit approval | Existing `kuairand-starter-kit/submit.py` |

## Experiment Lifecycle

Each experiment follows a small state machine:

```text
INSPECT
  -> HYPOTHESIZE
  -> RESERVE experiments/E####/spec.json
  -> VERIFY BOUNDARIES
  -> TRAIN
  -> SELECT BEST VALIDATION CHECKPOINT
  -> EVALUATE VALIDATION
  -> RUN SUBGROUP DIAGNOSTICS
  -> RECORD
  -> KEEP / REJECT / REFINE
```

Each experiment is a self-contained package. A pending package contains only
`spec.json`; execution adds `result.json` (or `failure.json`) and stores model
members under `checkpoints/` in the same directory. The append-only registry
remains the cross-experiment index.

One iteration changes exactly one feature family, loss decision, or modeling decision. The controller rejects an experiment if it attempts to:

- Modify a protected baseline, data, scoring, or published-score file.
- Select a model or checkpoint using the test split.
- Derive preprocessing state from validation or test data.
- Write outside the approved agent, candidate, experiment, run, or checkpoint roots.
- Exceed the six-hour or 50-iteration budgets.
- Generate a final submission without explicit approval.

The current `experiment_boundary.py` already provides protected hashes, editable roots, split enforcement, and initial budget constants. The controller should build on those rules rather than duplicate them.

## Data and Feature Architecture

`kuairand-starter-kit/data.py` remains the canonical source of split definitions, labels, and row order. It is hash-protected and should not be modified by experiments. Candidate feature builders wrap its output:

```text
data.py canonical rows
        |
        v
CandidateFeaturePipeline.fit(train)
        |
        +-- transform(train)
        +-- transform(valid)
        +-- transform(test only for final approved run)
```

For an event at time `t`, every aggregate must use only events before `t`. Validation and test row counts and ordering must never change. Same-row outcome fields such as `play_time_ms`, `is_click`, and `is_like` must not be predictors of `long_view`; they may be past-only historical aggregates or auxiliary training targets.

### Cleaning and feature ablation order

1. Give `duration_ms == 0` a dedicated missing-duration category instead of treating it as a genuinely short video.
2. Compare keeping all training rows, removing only exact duplicates, and inverse-frequency weighting of exact-duplicate groups.
3. Standardize field-specific missing categories without global replacement.
4. Add leakage-safe item popularity and smoothed item long-view rate.
5. Add user-author, user-tag, user-duration, and user-music affinities.
6. Add repeat-exposure count and time since the last video or author exposure.
7. Add hour bucket, weekday, weekend, recent activity, and video-age context.
8. Add recent-history summaries, embeddings, or attention only after simpler historical features are measured.

Static user metadata remains low priority. It is constant while ranking impressions for one user, so its first-order contribution cannot change the ranking. Existing ablations also found little gain from adding static user and item fields to the FM.

## Model and Loss Architecture

The current `baseline.py` already establishes the most useful first direction: hybrid BCE plus within-user BPR. BPR pairs must always contain a positive and negative impression from the same user.

The initial structured search should tune:

| Parameter | Initial values |
|---|---|
| BPR weight | `0.1, 0.25, 0.5, 1.0` |
| Learning rate | `3e-4, 1e-3, 3e-3` |
| Negatives per positive | `1, 4, 8` |

Hold embedding dimension `k=16`, batch size, and regularization fixed initially. After choosing the loss configuration, investigate negative-sampling strategy, embedding dimension, and L2 regularization. Promote the top three configurations only after evaluation across at least three random seeds.

Recommended progression:

1. Official pointwise BCE FM as the frozen reference.
2. Hybrid BCE plus within-user BPR.
3. Uniform versus mixed uniform/hard-negative sampling.
4. LambdaRank or LambdaLoss if GAUC improves while nDCG@5 stagnates.
5. Sequence-derived representations if historical affinity features show value.
6. Multitask learning or censored watch-time modeling as a later research phase.

Avoid increasing model capacity until loss alignment, historical features, and negative sampling have been tested. Existing experiments indicate that larger FM embeddings and additional static fields are not the main bottlenecks.

## Evaluation and Diagnostics

`kuairand-starter-kit/evaluate.py` remains the sole authority for GAUC, nDCG@5, and the primary score:

```text
primary = 0.5 * (GAUC + nDCG@5)
```

Every run records:

- Experiment ID, hypothesis, and exact specification.
- Git revision and code-diff hash.
- Dataset and protected-file hashes.
- Runtime environment and dependency versions.
- Feature set, model, loss, sampling method, and hyperparameters.
- Random seed and training budget.
- Epoch-level loss and validation metrics.
- Best checkpoint and best epoch.
- Validation GAUC, nDCG@5, and primary score.
- Three-seed mean and standard deviation for finalists.
- Subgroup metrics and regressions.
- Runtime, resource usage, errors, and timeouts.
- The orchestrator's conclusion: keep, reject, or refine.

Diagnostics should cover:

- Missing, short, medium, and long duration.
- Tab and calendar day.
- Frequent versus infrequent users.
- Popular versus rare videos.
- Seen versus unseen authors or items.
- Repeat versus first-time exposure.
- All-negative, mixed-label, and all-positive users.
- Earlier versus later validation periods to detect temporal drift.
- Standard versus random-exposure data when an unbiased diagnostic is appropriate.

## Promotion, Submission, and Stopping Rules

A candidate is promoted only if:

- Validation primary improves by at least `0.002`.
- The improvement survives at least three random seeds.
- No important subgroup collapses.
- The exact result is reproducible from the recorded specification.
- Protected-file, syntax, unit-test, and submission-alignment checks pass.

The leaderboard is an occasional diagnostic, not the optimization target. Do not submit every experiment.

The autonomous loop stops when no validation-primary improvement greater than `0.002` occurs for three consecutive well-motivated iterations, with finalists evaluated across at least three seeds. At that point, either accept the best candidate or begin a larger methodological change such as LambdaRank, sequence history, or multitask learning.

The test split is evaluated only for the final selected model after human approval. Final CSV generation and validation remain deterministic through `submit.py`.

## Implementation Phases

### Phase 0: Reproduction foundation

- Add a regression test for the official FM logits path.
- Verify that repeated calls do not mutate embeddings or cached intermediates.
- Correct the official pointwise FM logits with non-mutating dot products while
  preserving the organizer's model and training configuration.
- Add a deterministic, validation-only reproduction command for the official FM.
- Require validation scores to remain within `0.002` of the published baseline.
- Keep hidden-test data inaccessible throughout reproduction and experimentation.
- Confirm random and popularity sanity scores.

### Phase 1: Experiment spine

- **Complete:** define and validate the JSON experiment schema.
- **Complete:** implement the deterministic runner and controller.
- **Complete:** add checkpoint management and an append-only run registry.
- **Complete:** add boundary, budget, timeout, and validation-only selection checks.
- **Complete:** compare candidates against the protected published baseline.
- **Complete:** allocate experiment IDs without overwriting existing work.
- **Complete:** require a human approval receipt before final test evaluation and submission.
- **Complete:** begin with approved experiment templates so the first research
  cycle cannot execute arbitrary model-generated code. Phase 4 extends this
  foundation with governed pipeline operators and a sandboxed candidate-code
  path while retaining the same protected-file and hidden-test boundaries.

### Phase 2: EDA and diagnostics

- **Complete:** add deterministic dataset profiling and data-quality checks.
- **Complete:** add subgroup metric computation.
- **Complete:** generate structured baseline-versus-candidate comparisons.
- **Complete:** convert common diagnostic patterns into suggested next experiment families.

### Phase 3: First research cycle

- **Complete:** move the current BPR ensemble into the candidate architecture without changing its behavior.
- **Complete:** run the bounded BPR-weight, learning-rate, negative-count, embedding-size, and ensemble search.
- **Complete:** re-evaluate the strongest configurations across three seeds.
- **Complete:** store specifications, results, evidence, and decisions in the registry.
- **Outcome:** no candidate exceeded the protected published validation primary of `0.6016`; the best candidate reached `0.60114`.

### Phase 4: Governed full-stack autonomy

Phase 4 expands the agent from scalar tuning within two BPR templates into a
single-agent research loop that can choose meaningful changes across cleaning,
features, loss, model, and training strategy. The expansion remains governed:
metric calculation, experiment state, test access, budgets, and finalization
stay deterministic and outside the agent.

#### 4A: Typed pipeline proposals

- Replace the current `template + scalar parameters` proposal with a typed
  contract containing the target pipeline stage, hypothesis, evidence from
  prior diagnostics and experiments, transformation or experiment family,
  bounded parameters, expected effect, and seed.
- Allow the target stage to be one of `cleaning`, `features`, `loss`, `model`,
  or `training`; reject unknown stages and transformations before execution.
- Preserve the one-change rule: each iteration changes one feature family,
  cleaning decision, loss decision, model decision, or training decision.
- Include available operators, prior outcomes, convergence state, and subgroup
  regressions in the agent context so the next proposal is evidence-driven.

#### 4B: Reviewed operator catalogue

- Add approved experiment families for missing-duration handling, duplicate
  handling and weighting, item popularity and smoothed engagement rates,
  user-author/tag/duration/music affinities, repeat exposure and recency, time
  and distribution-shift features, and loss or sampling strategy.
- Connect `candidates/history_features.py` to experiment specifications and the
  runner so the agent can select leakage-safe historical features rather than
  merely discuss them.
- Run cleaning and feature ablations in the order defined by the data and
  feature architecture, using subgroup diagnostics to choose the next family.
- Require every operator to fit state on training data only, preserve row order
  and count, and pass label-invariance and test-sealing checks.

#### 4C: Sandboxed candidate-code changes

- After the reviewed operator loop is reliable, allow the agent to propose a
  source patch only under `candidates/`. A proposal records the patch, rationale,
  affected contract, and content hash; experiment JSON continues to contain no
  executable commands or arbitrary Python paths.
- Apply and execute candidate patches in an isolated temporary worktree or
  equivalent disposable workspace. Never give this path permission to change
  `data.py`, `evaluate.py`, `official_baseline.py`, `baseline_scores.json`, the
  controller, approval logic, registry, or prior experiment artifacts.
- Before training, require path validation, syntax and import checks, static
  policy checks, leakage tests, row-alignment tests, protected hashes, and the
  full automated test suite. Reject the patch atomically if any gate fails.
- Promote a successful candidate implementation into the reviewed catalogue
  only with its exact diff, tests, specification, and result evidence attached.

#### 4D: Autonomous reflection, recovery, and evidence

- Let the agent inspect each result, decide `keep`, `reject`, `refine`, or
  `change_direction`, and continue until the convergence or budget rule stops
  the research window.
- Add bounded recovery policies for invalid proposals, code errors, failed
  checks, training errors, and timeouts. Recovery may repair, retry once, or
  route to another approved direction without silently weakening a safety gate.
- Record configuration and code diffs, agent reasoning, input and output token
  counts, total agent wall-clock, training time, GPU-hours, failures, recovery
  events, and manual interventions in the append-only run evidence.

#### Phase 4 completion criteria

Phase 4 is complete when the agent can, without routine human intervention:

1. Inspect dataset, experiment, and subgroup evidence.
2. Choose a pipeline stage and form a specific hypothesis.
3. Configure an approved operator or propose one bounded candidate patch.
4. Pass boundary, leakage, reproducibility, and test checks.
5. Train and evaluate using validation only.
6. Interpret the result and subgroup effects.
7. Recover from a representative proposal or execution failure.
8. Select and run the next experiment until convergence.
9. Produce a complete audit summary including diffs, resources, recoveries, and
   the number of manual interventions.

### Phase 5: Advanced ranking

- Add LambdaRank only if diagnostics show weak top-five ordering.
- Add recent-history or attention models only after simpler affinity features demonstrate value.
- Add multitask or watch-time modeling only after the deterministic research loop is mature.

### Phase 6: Optional multi-agent expansion

Worker agents may later be introduced for bounded, independent tasks:

- Research one approved modeling direction.
- Inspect one distribution and report findings.
- Implement one approved experiment in an isolated worktree.
- Review one diff and run its verification suite.

The orchestrator remains the only owner of experiment state, budgets, convergence, checkpoint selection, and final submission. Model training itself is never an agent.

## Immediate Milestone

Phases 0, 1, 2, and 3 are complete. The next milestone is Phase 4A and 4B:
expand the proposal contract, connect reviewed cleaning and leakage-safe
historical-feature operators to the runner, and let the agent select them from
Phase 2 evidence. Sandboxed candidate-code generation follows only after that
operator loop passes its safety and recovery checks.
