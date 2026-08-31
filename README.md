# Autonomous ML Agent for KuaiRand-Pure

## Project overview

This project implements a governed autonomous machine-learning workflow for
the KuaiRand-Pure long-view ranking benchmark. The agent proposes bounded,
evidence-backed experiments, runs them using the training split and public
validation feedback only, records every decision, and handles selected runtime
and contract failures without overwriting prior evidence.

The final Phase 8 model is a three-seed sequence ensemble (E0016). It achieved
`0.604170` validation primary and `0.597778` final-test primary, compared with
the official FM baseline's `0.601600` validation and `0.594600` test primary.
The submitted model improves test GAUC by `+0.003927`, nDCG@5 by `+0.002430`,
and primary by `+0.003178`.

The final reproducibility package is in
[`kuairand-starter-kit/deliverables/phase8/`](kuairand-starter-kit/deliverables/phase8/),
including the submission CSV, all three final checkpoints, experiment evidence,
approval receipt, and detailed iteration/resource report.

## Final deliverables

- [Phase 8 results and iteration report](kuairand-starter-kit/deliverables/phase8/README.md)
- [Final E0016 submission CSV](kuairand-starter-kit/deliverables/phase8/E0016-submission.csv)
- [Final test metrics and submission provenance](kuairand-starter-kit/deliverables/phase8/final-result.json)
- [Validation evidence and three-seed member metrics](kuairand-starter-kit/deliverables/phase8/result.json)
- [Immutable final experiment specification](kuairand-starter-kit/deliverables/phase8/spec.json)
- [Human finalization approval receipt](kuairand-starter-kit/deliverables/phase8/approval.json)
- [Final model checkpoints](kuairand-starter-kit/deliverables/phase8/checkpoints/)

## Setup and installation

Requirements: Python 3.9+ and NumPy. The autonomous proposal loop additionally
uses Vertex/Gemini credentials and the optional packages in
`requirements-agent.txt`.

```bash
git clone <repository-url> auto-ml
cd auto-ml/kuairand-starter-kit

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy

# Optional: required only to run new Gemini-driven research campaigns.
python3 -m pip install -r requirements-agent.txt
```

Download KuaiRand-Pure so that the data directory is
`kuairand-starter-kit/KuaiRand-Pure/data`:

```bash
cd /path/to/auto-ml/kuairand-starter-kit
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Reproduce and verify the results

From `kuairand-starter-kit/`, first verify the protected baseline boundary and
the test suite:

```bash
python3 experiment_boundary.py --check
python3 -m unittest discover -s tests -p 'test_*.py'
```

Inspect the finalized metrics and verify the submission schema/alignment:

```bash
cat deliverables/phase8/final-result.json
python3 submit.py deliverables/phase8/E0016-submission.csv --check
```

The final bundle contains exactly the frozen E0016 experiment specification,
three checkpoint members, validation result, final-test result, and approval
receipt. Its SHA-verified submission is ready to submit as-is.

To reproduce the public validation baseline (without reading test labels):

```bash
python3 -m experiment_engine.reproduce_baseline
```

To start a fresh autonomous campaign, configure valid Vertex credentials and
run the campaign supervisor. This makes live Gemini API calls and therefore
incurs Vertex usage; its search trajectory may differ from Phase 8.

```bash
python3 -m experiment_engine.controller --campaign new-phase init-campaign
python3 -m agent.research --campaign new-phase \
  --url https://arxiv.org/abs/1808.09781 \
  --title "Self-Attentive Sequential Recommendation" \
  --summary "Causal self-attention over interaction history."
python3 -m agent.supervisor --campaign new-phase run --execute --max-steps 50
```

Candidate selection must remain validation-only. Do not run finalization during
a fresh search. Test evaluation is deliberately human-gated and may be run once
only after approving a replicated finalist.

## Limitations and future work

- The final model uses a compact NumPy implementation and an eight-item
  positive-history window, not a full transformer with positional or
  multi-interest modeling.
- E0008 was the best individual validation run, but E0016 was selected for its
  three-seed replication evidence. More repeated seeds and confidence intervals
  would better quantify variance.
- The search was limited to 15 recorded iterations, below the 50-iteration
  budget. More time could explore sequence-aware ensembles, temporal sequence
  features, and calibrated blending while preserving matched controls.
- The agent's proposal quality and cost depend on the external Gemini service;
  a local deterministic policy fallback could make long runs less dependent on
  network and credentials.
- Results are established on KuaiRand-Pure only. KuaiRand-1k and KuaiRand-27k
  were not attempted, so cross-dataset generalization remains unverified.
