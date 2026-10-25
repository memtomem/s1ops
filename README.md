# `s1ops` — Dedicated System 1 LLMOps Platform & SDK

> **The dedicated System 1 LLMOps SDK for sub-100ms reflexive inference (`noul` / `choice` / `score`).**
>
> **Install**: `pip install s1ops` · **Python**: 3.11+, `import s1ops` / `import s1` · **CLI**: `s1` / `s1ops`

`s1ops` is an offline-first, vendor-neutral evaluation, calibration, threshold compilation, blind label audit, and observability platform purpose-built for **Typed Decision Models (System 1)**:
- **TypeSafe `/v1/systemone`**: `jev-1.13.0`, `kev-1.0`, `laya`, `von` (`noul`, `choice`, `score`)
- **OpenAI `/v1/decisions`**: `gpt-6-luna` (`predicate`, `choice`)
- **Local / Custom Heads**: DeBERTa, ModernBERT, BGE rerankers, and constrained logprob classifiers emitting normalized probability distributions.

---

## Why System 1 Models Need Dedicated LLMOps

Traditional LLMOps platforms (**Langfuse**, **Arize Phoenix**, **LangSmith**) are architected for **System 2 generative workflows** (free-text completions, prompt templates, multi-turn RAG chains, token streaming latency). When applied to **System 1 decision models**, seven critical gaps emerge:

| Dimension | System 2 LLMOps (Langfuse / Phoenix) | `s1ops` (System 1 LLMOps) |
| --- | --- | --- |
| **Primary Output** | Unstructured text (`string`) | Calibrated probability vector (`noul`: `p` in `[0, 1]`; `choice`: a distribution over K options; `score`: ordinal `0..L-1`) |
| **Aggregation Unit** | Trace / Prompt Name | Strict **`(question_id, model_served, primitive)`** group — never pools across wordings or models |
| **Calibration & Small-n Bias** | None (only scalar scores) | **Equal-width/mass ECE**, **Monte Carlo Calibrated Noise Floor (`p95`)**, **Tie-Mass rounding detector**, **Temp/Platt/Isotonic** post-hoc fit |
| **Decision Thresholds & Routing** | Hardcoded `0.5` in app code | **Tie-safe Error-Budget Threshold (`select_threshold` + Clopper-Pearson)**, **System 1 → System 2 Escalation Band (`[lo, hi)`)**, & **`DecisionGate` Runtime Evaluator** |
| **Prompt / Question QA** | Free-form prompt playground | **Static Linter (`JL000`–`JL060`)** + **Probe Coherence** (Noul vs Yes/No Choice twin gap & inverse sum `abs(p + p_inv - 1)`) |
| **Distribution & Model Drift** | Token/latency or generic embedding drift | **Label-Free Probability PSI & Jensen-Shannon Divergence**, **Silent `model_served` Upgrade Alert**, & **Escalation Band Rate Shift** |
| **Data Privacy Default** | Logs full prompt/response bodies by default | **`safe-v1` Zero-State Default**: never stores raw item `state` in records, ledgers, or exported spans/scores |

`s1ops` bridges both worlds: you run rigorous System 1 calibration, linting, drift monitoring, and policy compilation locally (`s1` CLI + `s1 web`), execute sub-millisecond runtime gating via `s1.DecisionGate`, and export **`safe-v1` body-free scores, annotation queues, and OpenInference spans** directly into **Langfuse** and **Arize Phoenix**.

---

## Architecture Overview

**Inputs** (local files, 100% offline): `questions.json`; decision records `records.jsonl` / `records_canary.jsonl` (`jev-record/1`, `s1-record/1`); pre-declared slice features `features.jsonl`; an optional local state store `states_local.jsonl`.

**Core engine** (`import s1ops` / `import s1`):

1. **Lint & Probe Runner** — rules `JL000`–`JL060`, budget-guarded probe runner, Noul/Choice twin and inverse coherence.
2. **Calibration & Noise Floor** — log loss, Brier, ECE (equal-width and equal-mass), Monte Carlo noise floor, tie-mass detector.
3. **Improvement Compiler** — fit/apply (temperature, Platt, isotonic), error-budget threshold and escalation band tuner, $/1k cost simulator.
4. **Slice & Blind Label Audit** — answerability (Wilson CI), Benjamini-Hochberg FDR error slice miner, blind review sheet with control lift.
5. **Shadow & Judge Audit** — Noul `|Δp|` and Choice TVD, exact McNemar-Holm test, rubric quadratic-weighted κ and bias.
6. **Drift Monitor & Runtime Gate** — PSI and Jensen-Shannon drift, silent `model_served` guard, sub-millisecond `DecisionGate`.

**Outputs**: compiled `fits.json` and `policy.json`, a blind review sheet, `safe-v1` Langfuse runs/scores/queues, Phoenix / OpenInference `GUARDRAIL` / `EVALUATOR` spans, and a local 6-tab web workbench on `127.0.0.1:8765`. The state store is joined only locally, for the blind review sheet.

---

## Quickstart

### 1. Installation

```bash
pip install s1ops                # or: uv add s1ops
pip install "s1ops[langfuse]"    # optional: the Langfuse client for live upload
pip install "s1ops[typesafe]"    # optional: live `s1 probe --backend systemone`
pip install "s1ops[openai]"      # optional: live `s1 probe --backend openai`
```

### 2. Generate the Deterministic Synthetic Demo Workspace

```bash
s1 demo s1-demo
```

`s1 demo` writes a complete, seeded offline workspace into a new (or empty) directory and refuses anything else, so it never overwrites a real workspace. `--seed N` changes the generated data.
- `questions.json`: Canonical questions + intentional `JL010` / `JL011` / `JL014` / `JL023` lint cases.
- `records_train.jsonl`: 120 calibration-fold records (`jev-1.13.0`).
- `records.jsonl`: 160 held-out evaluation records (`jev-1.13.0`, with calibrated probabilities from `fits.json`, 6 planted gold-label errors, and a concentrated failure slice on `lang=ko_slang`).
- `records_kev.jsonl`: 160 paired shadow records (`kev-1.0`) on the same items.
- `records_canary.jsonl`: 120 production canary records simulating distribution drift (`PSI > 0.25`) + silent backend upgrade on a third of traffic (`jev-1.13.0` → `jev-1.14.0-rc1`).
- `probe_records.jsonl`: Noul vs Yes/No Choice twin & inverse polarity probes.
- `features.jsonl`: Pre-declared slice tags (`answerable`, `unanswerable`, `adversarial_trap`, `lang`, `has_code`).
- `states_local.jsonl`: Local-only synthetic item states for the blind review sheet.
- `judge_grades.jsonl`: 90 paired rubric judge vs human grades across 3 criteria (`faithfulness`, `policy_compliance`, `tone`).
- `fits.json` + `policy.json`: A compiled calibration fit and routing policy.

### 3. Run the End-to-End CLI Pipeline

```bash
# 1) Static Question Lint & Probe Coherence Audit
s1 lint rules
s1 lint check s1-demo/questions.json
s1 lint probe-audit s1-demo/probe_records.jsonl

# 2) Post-Hoc Calibration Fit, Out-of-Sample Apply, and Noise-Floor Audit
s1 fit s1-demo/records_train.jsonl -o s1-demo/fits.json --method temperature
s1 apply s1-demo/records.jsonl --fit s1-demo/fits.json -o s1-demo/records_calibrated.jsonl
s1 audit s1-demo/records.jsonl --fit s1-demo/fits.json

# 3) Error-Budget Threshold & System 1->2 Escalation Band Compilation
s1 policy tune s1-demo/records.jsonl -o s1-demo/policy.json --error-budget 0.10 --use-calib
s1 policy curve s1-demo/records.jsonl --use-calib

# 4) Answerability Pre-Flight & Benjamini-Hochberg FDR Error Slice Mining
s1 slice answerable s1-demo/records.jsonl --features s1-demo/features.jsonl
s1 slice mine s1-demo/records.jsonl --features s1-demo/features.jsonl --fdr-q 0.10

# 5) Calibrated Label Disagreement Ranking & Blind Review Sheet
s1 labelaudit rank s1-demo/records.jsonl --fit s1-demo/fits.json --top 10
s1 labelaudit sheet s1-demo/records.jsonl \
  --states s1-demo/states_local.jsonl \
  --fit s1-demo/fits.json \
  -o s1-demo/blind_sheet.md \
  -k s1-demo/blind_key.json \
  --top 6 --controls 6

# 6) Multi-Backend Shadow Compare (McNemar-Holm) & Rubric Judge Agreement (Weighted κ)
s1 compare s1-demo/records.jsonl s1-demo/records_kev.jsonl
s1 judge audit s1-demo/judge_grades.jsonl

# 7) Distribution & Silent Model-Served Drift Monitor
s1 drift s1-demo/records.jsonl s1-demo/records_canary.jsonl --policy s1-demo/policy.json

# 8) Budget-Guarded Probe Runner (built-in deterministic mock backend) & Runtime DecisionGate
s1 probe s1-demo/questions.json --states s1-demo/states_local.jsonl \
  -o s1-demo/probe_live.jsonl --ledger s1-demo/probe_ledger.jsonl --cap-usd 0.50
s1 gate s1-demo/policy.json --fit s1-demo/fits.json \
  --question q-30076be2728be682 --model jev-1.13.0 --p 0.52

# 9) Safe-v1 Body-Free Export to Langfuse & Arize Phoenix (OpenInference)
s1 export langfuse s1-demo/records.jsonl -o s1-demo/langfuse_export.json --policy s1-demo/policy.json
s1 export phoenix s1-demo/records.jsonl -o s1-demo/phoenix_spans.jsonl --policy s1-demo/policy.json
```

**Live probes.** `s1 probe --backend systemone` sends each state and question to `POST /v1/systemone` (key in `TYPESAFE_API_KEY`, endpoint in `TYPESAFE_BASE_URL` or `--base-url`); `--backend openai` sends them to OpenAI's `POST /v1/decisions` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, model `gpt-6-luna` by default) with prompt caching disabled, so the measured cost is exact. A question that API cannot take as written — a Noul with `criteria`, structured instructions, or option and level counts outside its limits — is refused before anything is reserved, rather than sent in a different form under the same `question_id`. A refused answer is recorded with `error_code: "refusal"`. The APIs report billable input tokens, not dollars, so `--usd-per-input-token` is required. Every attempt reserves its worst case first (`--worst-usd-per-token`, plus `--reserve-overhead-tokens` for server-side prompt tokens) and is settled as measured, or, when the request may have been billed without a usable reported cost (timeout, connection error, HTTP error), at its full reservation. Timeouts, connection errors, 408, 429 and 5xx are retried up to `--max-retries` times. An item that still fails is written as a record with `model_served: null` and an `error_code`, which `s1 drift` counts as a failure. Records are written as they arrive, so a run stopped by `--cap-usd` or by a measured cost above its reservation keeps what it paid for. Error messages carry only an error code, never server text, since a server may echo the request.

### 4. Launch the 6-Tab Local Web Workbench

```bash
s1 web --workspace s1-demo --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` to explore:
1. **Calibration & Noise Floor**: SVG Reliability Diagram (Before vs After Fit), Monte Carlo 95th-percentile ECE Noise Floor, and Tie-Mass rounding detector.
2. **Threshold & Escalation Simulator**: Interactive `band_lo`, `band_hi`, `cost_sys2`, and `error_budget` sliders with live System 1 Auto vs System 2 Escalation split and Pareto frontier table.
3. **Question Lint & Probe Coherence**: Static `JL000`–`JL060` diagnostics + Noul/Choice twin gap and polarity inverse sum audit.
4. **Slices & Blind Label Audit**: Wilson 95% CI Answerability table, Benjamini-Hochberg FDR (`q <= 0.10`) error slices, and top calibrated label disagreement candidates.
5. **Shadow Compare & Judge Audit**: Paired `jev-1.13.0` vs `kev-1.0` accuracy, flip rate, exact McNemar Holm-adjusted p-value badges, and per-criterion quadratic-weighted κ.
6. **Distribution & Model Drift**: Baseline vs Canary PSI, Jensen-Shannon Divergence, mean/entropy/escalation rate shift, and silent `model_served` upgrade alerts (`CRITICAL` / `WARN` / `OK`).

---

## Python SDK Usage (`import s1` / `import s1ops`)

The outputs in the comments are for a fresh `s1 demo s1-demo` workspace; the CLI pipeline above rewrites `fits.json` and `policy.json`, which changes the gate's decision.

```python
import s1

# 1. Define a question and compute its canonical wire-form question_id
q = s1.NoulQuestion(instructions="Does the user message request a financial wire transfer?")
qid = s1.question_id(q)  # -> 'q-30076be2728be682'

# 2. Load and strictly group decision records by (question_id, model_served, primitive);
#    the demo has one noul group per file
train = s1.select(s1.read_records("s1-demo/records_train.jsonl")).groups[0]
group = s1.select(s1.read_records("s1-demo/records.jsonl")).groups[0]

# 3. Fit temperature scaling on the calibration fold, then audit the held-out fold
#    before/after calibration against the Monte Carlo noise floor
fit = s1.fit_group(train, method="temperature", split="cal")
after_P = s1.transform(fit.method, fit.primitive, fit.params, group.P)
report = s1.audit_group(group, bins=10, reps=500, after=after_P)
print("Raw ECE (mass):", report["before"]["ece_mass"], "-> Calibrated:", report["after"]["ece_mass"])

# 4. Sub-millisecond Runtime Gating (DecisionGate)
gate = s1.DecisionGate.from_files("s1-demo/policy.json", "s1-demo/fits.json")
decision = gate.evaluate_noul(
    question_id=group.question_id,
    model_served=group.model_served,
    p_raw=0.52,
)
print(decision.action, decision.escalated, decision.p_calibrated)  # -> 'ESCALATE', True, ...
```

---

## Safety & Statistical Guardrails

- **Zero-State Privacy (`safe-v1`)**: `s1-record/1` files, budget ledgers, probe outputs, and Langfuse/Phoenix exports never store raw item `state` text.
- **No Cross-Group Pooling**: Metrics, fits, drift audits, and thresholds are keyed strictly to `(question_id, model_served, primitive)`.
- **Out-of-Sample & Model-Mismatch Enforcement (`s1 apply` & `DecisionGate`)**:
  - Attempting to apply a calibration fit to any `(dataset, item_id)` used during `s1 fit` exits with **code `2`**.
  - Attempting to apply a calibration fit or runtime `DecisionGate` to a record whose `model_served` differs from the fitted/compiled model exits with **code `3`** (`ModelServedMismatchError`).
- **Policy Fingerprint (`policy_sha256`)**: every `PolicyEntry` carries a canonical SHA-256 fingerprint of its routing fields. Verification is opt-in: pass the expected fingerprint to `DecisionGate.evaluate_noul(expected_policy_sha256=...)` or `s1 gate --expected-sha256`, and a mismatch raises `PolicyIntegrityError`. Loading `policy.json` recomputes the fingerprint; it does not compare against the stored value.
- **Reserve-Then-Settle Budget Ledger (`s1 ledger` & `s1 probe`)**: Uses exclusive `flock` on `<journal>.lock`, rejects hard-linked ledgers, and halts immediately if a measured cost exceeds its reservation (`LedgerUnusable`).

---

## CLI Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Usage error, missing or malformed input, no matching groups, `lint check --strict` findings, policy fingerprint mismatch, ledger error, `s1 demo` target not new or empty |
| `2` | `s1 apply` refused an in-sample item (override with `--allow-in-sample`) |
| `3` | `model_served` mismatch against the fit or policy (`s1 apply`, `s1 gate`) |
| `4` | `s1 drift --fail-on-critical` found a `CRITICAL` question |

Errors print a single `s1: <message>` line on stderr. When an input that another command produces is missing, the message names that command (for example `run \`s1 fit\` first`). Set `S1_DEBUG=1` to re-raise errors raised by the library, with their traceback; checks the CLI makes itself (a missing input file, an empty grades file) still print one line. A missing input path is an error everywhere, including `s1 ledger status`, except `s1 lint check`, which reports an unreadable questions file as a `JL000` finding in its normal (`--format json`) report.

---

## Development

From a source checkout (the sdist includes the test suite), all tests run 100% offline with zero network calls or API keys:

```bash
uv sync --all-extras
uv run pytest -v
```
