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
| **Aggregation Unit** | Trace / Prompt Name | Strict **`(question_id, model_served, primitive)`** group — never pools across wordings or models (drift's per-question report pools models on purpose, so a model change shows) |
| **Calibration & Small-n Bias** | None (only scalar scores) | **Equal-width/mass ECE**, **Monte Carlo Calibrated Noise Floor (`p95`)**, **Tie-Mass rounding detector**, **Temp/Platt/Isotonic** post-hoc fit |
| **Decision Thresholds & Routing** | Hardcoded `0.5` in app code | **Tie-safe Error-Budget Threshold (`select_threshold` + Clopper-Pearson)**, **System 1 → System 2 Escalation Band (`[lo, hi)`)**, & **`DecisionGate` Runtime Evaluator** |
| **Prompt / Question QA** | Free-form prompt playground | **Static Linter (`JL000`–`JL060`)** + **Probe Coherence** (Noul vs Yes/No Choice twin gap & inverse sum `abs(p + p_inv - 1)`) |
| **Distribution & Model Drift** | Token/latency or generic embedding drift | **Label-Free Probability PSI & Jensen-Shannon Divergence**, **Silent `model_served` Upgrade Alert**, & **Escalation Band Rate Shift** |
| **Data Privacy Default** | Logs full prompt/response bodies by default | **`safe-v1` Zero-State Default**: never writes raw item `state` into records, probe ledgers, or exported spans/scores |

`s1ops` bridges both worlds: you run rigorous System 1 calibration, linting, drift monitoring, and policy compilation locally (`s1` CLI + `s1 web`), execute sub-millisecond runtime gating via `s1.DecisionGate`, and export **`safe-v1` body-free scores, annotation queues, and OpenInference spans** directly into **Langfuse** and **Arize Phoenix**.

---

## Architecture Overview

**Inputs** (local files; everything except a live probe runs offline): `questions.json`; decision records `records.jsonl` / `records_canary.jsonl` (`jev-record/1`, `s1-record/1`); pre-declared slice features `features.jsonl`; an optional local state store `states_local.jsonl`.

**Questions file.** A JSON object mapping question names to questions, or that mapping under one key, `{"questions": {...}}`. `s1 lint check`, `s1 probe` and `s1 labelaudit sheet --questions` read it the same way. An entry that is not a question, a repeated name, an empty file, or nesting past 100 levels stops `s1 probe` and `s1 labelaudit sheet` before anything is sent or written; `s1 lint check` reports each as a `JL000` finding and exits 1.

**Fix the wording before probing.** `question_id` hashes the exact wording, option order included. Data that permutes a question's options per row gives every permutation its own `question_id`, and so its own fits and policies: pick one wording per question first.

**A rating is a judgment.** "How many stars did this reviewer give?" with levels `"1 star: very negative"` .. `"5 stars: very positive"` gets `JL012` (counting) and `JL031` (numeric levels), which are written for quantities the model would have to count. Suppress them on that question with `"x-s1-lint": {"ignore": ["JL012", "JL031"]}`; `question_id` does not change.

**Core engine** (`import s1ops` / `import s1`):

1. **Lint & Probe Runner** — rules `JL000`–`JL060`, budget-guarded probe runner, Noul/Choice twin and inverse coherence.
2. **Calibration & Noise Floor** — log loss, Brier, ECE (equal-width and equal-mass), Monte Carlo noise floor, tie-mass detector.
3. **Improvement Compiler** — fit/apply (temperature, Platt, isotonic), error-budget threshold and escalation band tuner, $/1k cost simulator.
4. **Slice & Blind Label Audit** — answerability (Wilson CI), Benjamini-Hochberg FDR error slice miner, blind review sheet with control lift.
5. **Shadow & Judge Audit** — Noul `|Δp|` and Choice TVD, exact McNemar-Holm test, rubric quadratic-weighted κ and bias.
6. **Drift Monitor & Runtime Gate** — PSI and Jensen-Shannon drift, silent `model_served` guard, sub-millisecond `DecisionGate`.

**Outputs**: compiled `fits.json` and `policy.json`, a blind review sheet, `safe-v1` Langfuse runs/scores/queues, Phoenix / OpenInference `GUARDRAIL` / `EVALUATOR` / `RERANKER` spans (noul / choice / score), and a local 6-tab web workbench on `127.0.0.1:8765`. The state store is read by two commands only: `s1 labelaudit sheet` writes the state text of every item under review (the top-ranked flags plus random controls) into the review sheet, which is meant for a human reviewer, and `s1 probe` sends each state to the selected backend (a live API for `--backend systemone` or `openai`). s1ops never writes that state text into records, ledgers, probe output or exports; `s1 apply`, which copies its input rows, leaves a top-level `state` out.

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
- `records_canary.jsonl`: 120 production canary records simulating distribution drift (`PSI > 0.25`) + silent backend upgrade on a third of traffic (`jev-1.13.0` → `jev-1.14.0-rc1`). The `jev-1.13.0` records carry `calib` from `fits.json`, as `s1 apply` writes it; the upgraded ones have no fit.
- `probe_records.jsonl`: Noul vs Yes/No Choice twin & inverse polarity probes.
- `features.jsonl`: Pre-declared slice tags (`answerable`, `unanswerable`, `adversarial_trap`, `lang`, `has_code`).
- `states_local.jsonl`: Synthetic item states for the blind review sheet and the probe runner. One row per item, `{"dataset", "item_id", "state"}`; both commands check rows the same way and read a row without `dataset` as `dataset: "default"`, the value `s1 probe` writes into its records. To match records that carry no `dataset`, write `"dataset": null`.
- `judge_grades.jsonl`: 90 paired rubric judge vs human grades across 3 criteria (`faithfulness`, `policy_compliance`, `tone`).
- `fits.json` + `policy.json`: A compiled calibration fit and routing policy.
- `records_categorical_train.jsonl` / `records_categorical.jsonl`: 150 + 150 choice (`support_intent`) and score (`urgency`) records per fold (`jev-1.13.0`, overconfident on purpose), the held-out fold with calibrated probabilities from `fits_categorical.json`.
- `fits_categorical.json` + `policy_categorical.json`: their temperature fits and error-budget policies, tuned on calibrated probabilities. The web workbench reads the noul files only.

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
# For a consumer that acts only on BLOCK: the smallest cutoff at which at most 2 % of label-false items pass
s1 policy tune s1-demo/records.jsonl -o s1-demo/policy_false_pass.json --max-false-pass 0.02 --use-calib
s1 policy curve s1-demo/records.jsonl --use-calib

# 3b) The same for choice and score answers
s1 fit s1-demo/records_categorical_train.jsonl -o s1-demo/fits_categorical.json --method temperature
s1 apply s1-demo/records_categorical.jsonl --fit s1-demo/fits_categorical.json -o s1-demo/records_categorical_calibrated.jsonl
s1 audit s1-demo/records_categorical.jsonl --fit s1-demo/fits_categorical.json
s1 policy tune s1-demo/records_categorical.jsonl -o s1-demo/policy_categorical.json --error-budget 0.10 --use-calib

# 4) Answerability Pre-Flight & Benjamini-Hochberg FDR Error Slice Mining
s1 slice answerable s1-demo/records.jsonl --features s1-demo/features.jsonl
s1 slice mine s1-demo/records.jsonl --features s1-demo/features.jsonl --fdr-q 0.10

# 5) Calibrated Label Disagreement Ranking & Blind Review Sheet
s1 labelaudit rank s1-demo/records.jsonl --fit s1-demo/fits.json --top 10
#    (a sheet asks one question: pass --question when the records hold several; its options are the
#    question's classes, and --options may only reorder them; --questions prints the wording)
s1 labelaudit sheet s1-demo/records.jsonl \
  --states s1-demo/states_local.jsonl \
  --questions s1-demo/questions.json \
  --fit s1-demo/fits.json \
  -o s1-demo/blind_sheet.md \
  -k s1-demo/blind_key.json \
  --top 6 --controls 6

# 6) Multi-Backend Shadow Compare (McNemar-Holm) & Rubric Judge Agreement (Weighted κ)
s1 compare s1-demo/records.jsonl s1-demo/records_kev.jsonl
s1 judge audit s1-demo/judge_grades.jsonl

# 7) Distribution & Silent Model-Served Drift Monitor (PSI on raw probabilities, alerting only when a permutation test finds it significant;
#    --calibrated compares calibrated ones and needs `calib` on every answered record of both files, from one fit per question and model;
#    --raw-only, the 0.1.0 name of the default, cannot be combined with it; a stored `calib` is validated either way)
s1 drift s1-demo/records.jsonl s1-demo/records_canary.jsonl --policy s1-demo/policy.json
#    (a live canary has no `calib` yet: pass `--fit s1-demo/fits.json`, which calibrates each file in memory, as
#    `s1 apply canary.jsonl --fit s1-demo/fits.json -o canary_calibrated.jsonl --pass-through-unfitted` writes it,
#    keeping the records of models without a fit)

# 8) Budget-Guarded Probe Runner (built-in deterministic mock backend) & Runtime DecisionGate
s1 probe s1-demo/questions.json --states s1-demo/states_local.jsonl \
  -o s1-demo/probe_live.jsonl --ledger s1-demo/probe_ledger.jsonl --cap-usd 0.50
s1 gate s1-demo/policy.json --fit s1-demo/fits.json \
  --question q-30076be2728be682 --model jev-1.13.0 --p 0.52
# a choice or score question takes its distribution instead of --p
s1 gate s1-demo/policy_categorical.json --fit s1-demo/fits_categorical.json \
  --question q-a70bd97f0b7f1055 --model jev-1.13.0 \
  --probs '{"billing": 0.97, "technical": 0.02, "account": 0.01}'   # SELECT:billing
s1 gate s1-demo/policy_categorical.json --fit s1-demo/fits_categorical.json \
  --question q-5c05e56162b3eaa0 --model jev-1.13.0 \
  --probs '{"0": 0.1, "1": 0.45, "2": 0.4, "3": 0.05}'   # ESCALATE

# 9) Safe-v1 Body-Free Export to Langfuse & Arize Phoenix (OpenInference)
s1 export langfuse s1-demo/records.jsonl -o s1-demo/langfuse_export.json --policy s1-demo/policy.json
s1 export phoenix s1-demo/records.jsonl -o s1-demo/phoenix_spans.jsonl --policy s1-demo/policy.json
s1 export phoenix s1-demo/records_categorical.jsonl -o s1-demo/phoenix_categorical.jsonl --policy s1-demo/policy_categorical.json
```

**Live probes.** `s1 probe --backend systemone` sends each state and question to `POST /v1/systemone` (key in `TYPESAFE_API_KEY`, endpoint in `TYPESAFE_BASE_URL` or `--base-url`); `--backend openai` sends them to OpenAI's `POST /v1/decisions` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`) with prompt caching disabled. Both need `--model`; only the offline mock has a default. A response that reports a cache write has no known price and is settled at its full reservation; cached reads, which are discounted, are charged at the full input price, an upper bound. A question that API cannot take as written — a Noul with `criteria`, object (rather than string or input-part list) instructions, or option and level counts outside its limits — is refused before anything is reserved, rather than sent in a different form under the same `question_id`. A refused answer is recorded with `error_code: "refusal"`. `/v1/systemone` resolves an alias to the version that answered (`jev-latest` came back as `jev-1.14.0`), so a silent upgrade shows in `model_served`; `/v1/decisions` returned the alias `gpt-6-luna` itself, so a model change behind it is visible only as distribution drift. `model_served` is whatever the server reports: a compatible server that echoes the request records the name you passed, and one that names every checkpoint alike leaves a checkpoint change visible only as distribution drift too, since drift's model comparison and the exit 3 of `s1 apply` and `s1 gate` compare names. The APIs report billable input tokens, not dollars, so `--usd-per-input-token` is required. Every attempt reserves its worst case first (`--worst-usd-per-token`, plus `--reserve-overhead-tokens` for the prompt tokens a server bills on top of the state and question: 300 for systemone and 200 for openai by default: a short state was billed up to 216 and 98 tokens more than estimated, and about 236 and 118 extrapolated to an empty one) and is settled as measured, or, when the request may have been billed without a usable reported cost (timeout, connection error, HTTP error), at its full reservation. Timeouts, connection errors, 408, 429 and 5xx are retried up to `--max-retries` times. An item that still fails is written as a record with `model_served: null` and an `error_code`, which `s1 drift` counts as a failure. Records are written as they arrive, so a run stopped by `--cap-usd` or by a measured cost above its reservation keeps what it paid for. Error messages carry only an error code, never server text, since a server may echo the request.

**Probe coherence.** Unless `--no-expand-probes` is given, `s1 probe` sends each Noul with string instructions, or with a supplied inverse (below), three ways: as written (`probe_role: "noul"`), as a yes/no Choice with the same instructions (`twin`), and as its inverse (`inverse`). `s1 lint probe-audit` then reports, per question and served model, the twin gap `|p - p_yes|` and the inverse error `|p + p_inv - 1|`. Write the inverse yourself when you can, under the question's lint extension:

```json
{"positive": {"type": "noul", "instructions": "Is this movie review positive?",
              "criteria": {"true": "positive", "false": "negative"},
              "x-s1-lint": {"inverse": {"instructions": "Is this movie review negative?",
                                        "criteria": {"true": "negative", "false": "positive"}}}}}
```

The extension is stripped before hashing, so the question keeps the `question_id` `s1 lint check` gives it. Without one, the inverse is generated as `"Is it false that: <instructions>"` with the true/false criteria swapped; a model that reads that sentence literally answers it like the original, so its error can measure the generated wording more than the model (on a local Kev, s1ops 0.6.0's generated inverse, which also dropped the criteria, gave an inverse flip rate of 1.0 next to a twin gap of 0.012). A Noul with object or list instructions and no supplied inverse is sent once, as written, and the audit skips it. Each inverse record says which wording it got (`probe_inverse`: `supplied` or `generated`), and the audit reports it as `inverse_source` (`null` for records from before the tag, or a mix). A question and model whose inverse records carry more than one `question_id` are refused rather than pooled, so audit the output of one wording at a time. A question carries `x-s1-lint` or `x-jev-lint`, not both, and `s1 probe` refuses a malformed extension, as `s1 lint check` reports it, before the ledger or the output is opened.

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
6. **Distribution & Model Drift**: Baseline vs Canary PSI with its permutation p-value and Jensen-Shannon Divergence on raw probabilities, mean/entropy/escalation rate shift, and silent `model_served` upgrade alerts (`CRITICAL` / `WARN` / `OK`).

---

## Python SDK Usage (`import s1` / `import s1ops`)

The outputs in the comments are for a fresh `s1 demo s1-demo` workspace; the CLI pipeline above rewrites `fits.json` and `policy.json`, which changes the gate's decision.

`import s1` resolves every export at once, the fitting stack (`scipy.optimize`) included. `import s1ops` resolves exports on first use, and a process that only gates, such as a per-call hook, can import the gate directly: `from s1ops.runtime.gate import DecisionGate` (with `load_policies` / `load_fits`) loads numpy, `scipy.special` and pydantic, not scipy's optimizers, scikit-learn, the CLI or the web app.

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

# Choice and score questions: evaluate_choice / evaluate_score take the distribution;
# the action is SELECT:<label> or ESCALATE, and a score decision also reports expected_score.
# gate.evaluate_choice(question_id=..., model_served=..., probs_raw={"A": 0.7, "B": 0.3})
# gate.evaluate_score(question_id=..., model_served=..., probs_raw={"0": 0.1, "1": 0.2, "2": 0.7})
```

---

## Safety & Statistical Guardrails

- **Zero-State Privacy (`safe-v1`)**: record files, the ledgers `s1 probe` writes, probe outputs, and Langfuse/Phoenix exports never store raw item `state` text. `s1 apply` copies its input rows, so it leaves a top-level `state` out of every row it writes (`state_sha256` and the other fields are kept) and adds `"state_dropped": N` to its JSON summary when any input record had one. `Ledger.reserve` / `settle` refuse a `state` metadata key (`ValueError`, before anything is written), and `s1 ledger recover` refuses a journal whose entries carry one rather than copy it into a new file; a journal written that way by s1ops 0.2.0 or earlier still opens.
- **One Routing Rule**: `DecisionGate`, `s1 drift`'s escalation rate, and the `policy_action` of both exporters apply a policy through the same rule. Every one of them routes the distribution the policy was tuned on: the calibrated one exactly when the policy was compiled with `--use-calib` (`calibrated: true`), whatever a record carries. The reports take it from the record's `calib`, so a calibrated policy needs `s1 apply` output, or `--fit` on `s1 drift` and `s1 export`, and refuses a record without `calib`; `s1 drift` applies to each record only the policy of its own `(question_id, model_served, primitive)` group, so records from another model (the demo canary's `jev-1.14.0-rc1` third) are left out of the escalation rate, which is `null` when no record has a policy. The distributions `s1 drift` compares (PSI, JS divergence, quantiles, entropy) are a separate choice: raw on both sides, or calibrated on both with `--calibrated`, which refuses an answered record without `calib` rather than compare it raw, and records of one `(question_id, model_served, primitive)` group calibrated through different fits (`calib.fit_sha256`), since a refit would show as drift; the report says which in `distribution`. s1ops 0.1.0 and earlier compared each record's `calib` when present and its raw probabilities otherwise, and their drift reports have no `distribution`. A Noul `error_budget` policy accepts when `max(p, 1 - p)` reaches the threshold (else escalates); a band accepts at or above `band_hi`, blocks below `band_lo`, and escalates between. `s1 policy tune --max-false-pass RATE` writes a Noul band with `band_lo == band_hi`, which never escalates: its cutoff is the smallest observed probability (or `1.0`) at which at most `RATE` of the group's label-false items reach it (with `--use-cp-upper`, its Clopper-Pearson upper bound), and `metrics_on_split` records the counts it was chosen on (every other policy records what it does on the split: `coverage`, `accepted`, `errors`, `accepted_error`); it replaces the F1 band, so `--max-escalation` is refused with it, and choice and score groups keep the error budget. A choice or score policy selects the top answer when its probability reaches the threshold, and escalates on a tie at the top or when no threshold met the error budget (`threshold: null`); a policy tuned with `--use-calib` must be gated with its `fits.json`. Exports of policy-routed records carry `routing_rule` (`s1.routing_rule` on Phoenix spans, `metadata.routing_rule` on Langfuse observations), currently `"3"`; exports from s1ops 0.1.0 and earlier have none, and their `policy_action` and `predicted` may differ for the same policy. Choice and score probabilities are renormalized with an exactly rounded sum, so retune choice or score policies compiled by 0.1.0 or earlier on Python 3.11 (run `s1 apply` first for `--use-calib` ones).
- **A Noul Band Is Not Bound by the Budget**: every policy records in `metrics_on_split` what it does on the split it was tuned on, by the routing rule itself. For a Noul group `s1 policy tune` picks an escalation band that maximizes F1 under `--max-escalation`, and the band then routes; `--error-budget` and `--use-cp-upper` govern the separate `threshold`, not the band. So `accepted_error` can exceed the budget: on the demo records, raw, 0.238 at a budget of 0.10. `--max-escalation` does not help: at 0 the band is empty and every item is accepted (0.275 there). `--max-false-pass` replaces the band with a cutoff that bounds the label-false items that pass.
- **No Cross-Group Pooling**: Metrics, fits, thresholds, and slice reports are keyed strictly to `(question_id, model_served, primitive)`; `s1 slice answerable` / `mine` and the workbench's Slices tab report each group separately. Two commands are exceptions. Drift reports per `question_id` and pools its models, so a changed `model_served` shows (CRITICAL); its policy lookup and calibrated fit check stay per group, and it refuses a question whose records name more than one primitive (a `question_id` names one wire question). `s1 compare` refuses a question whose paired records on one side come from more than one model or primitive; split such a file by `model_served`.
- **Drift Alerts Need Evidence (`s1 drift`)**: PSI of two samples from one distribution grows with smaller samples (about 0.33 at 54 records per side, over the `CRITICAL` threshold of 0.25), so a PSI over a threshold raises the status only when a permutation test puts it outside sampling noise: `CRITICAL` needs PSI >= 0.25 and p < 0.01, `WARN` needs PSI >= 0.10 and p < 0.05. The p-value (`psi_p_value`) is a Monte Carlo estimate from 1,000 resamples of the answered records' bin counts (seeded per question; the draws can differ under another numpy, which matters only for a p near alpha). A PSI over a threshold that is not significant is listed in `notes`, not `alerts`; each question also reports `n_ref_answered` / `n_cur_answered`, the records the test used. `OK` does not establish stability: small samples miss real shifts (in one measurement, 50 of 120 comparisons with a moderate shift at about 54 records per side stayed `OK`). The test treats records as exchangeable, so it does not fit autocorrelated windows, and its alpha holds per question and per comparison, not across questions or repeated runs. Model changes, `model_requested` mismatches and failure-rate jumps alert as before. `audit_drift(..., psi_resamples=0)` restores the threshold-only rule. s1ops 0.6.0 and earlier alerted on PSI alone.
- **Out-of-Sample & Model-Mismatch Enforcement (`s1 apply` & `DecisionGate`)**:
  - Attempting to apply a calibration fit to any `(dataset, item_id)` used during `s1 fit` exits with **code `2`**.
  - Attempting to apply a calibration fit or runtime `DecisionGate` to a record whose `model_served` differs from the fitted/compiled model exits with **code `3`** (`ModelServedMismatchError`). For a production file that mixes models, `s1 apply --pass-through-unfitted` calibrates the records it has a fit for, writes the others unchanged (except a top-level `state`, as above), and prints how many of each (with the reason per group); class mismatches and in-sample items are still refused.
  - `s1 drift --fit` and `s1 export langfuse|phoenix --fit` do the same in memory, so no intermediate file is needed: reports and export files equal the ones made through `s1 apply --pass-through-unfitted`, a stored `calib` is replaced on a record the fit file has a fit for and kept on any other, in-sample items exit `2` (`--allow-in-sample`), and class mismatches exit `1`. A refusal to apply the fit, and an input line the record reader refuses, name the input file and its own line. `s1 drift`'s report and the printed export summaries gain a `fit` block with what was calibrated and what was passed through, per file; an export file is unchanged. `s1 drift --fit` needs `--policy` or `--calibrated`, since a raw comparison without a policy compares and routes no calibrated probabilities. The web workbench does not take a fit. In the SDK: `apply_fits(read_records(path), fits, pass_through_unfitted=True).records`.
- **Policy Fingerprint (`policy_sha256`)**: every `PolicyEntry` carries a canonical SHA-256 fingerprint of its routing fields. Verification is opt-in: pass the expected fingerprint to `DecisionGate.evaluate_noul(expected_policy_sha256=...)` or `s1 gate --expected-sha256`, and a mismatch raises `PolicyIntegrityError`. Loading `policy.json` also compares each row's stored fingerprint with one recomputed from its fields and refuses a mismatch (`PolicyFingerprintError`, CLI exit 1, web 400), so a threshold or band edited after `s1 policy tune` wrote the file is caught at load. To accept a deliberate hand edit, remove that row's `policy_sha256`; a row without one is only recomputed.
- **Fit Provenance (`fit_sha256`)**: `s1 apply` writes the fit's `fit_sha256` into each record's `calib`. `s1 policy tune --use-calib` refuses a selected record without calibrated probabilities, a non-null `calib.fit_sha256` that is not a string, and a group whose records were calibrated through different fits, and records the group's hash in the policy (`null` when every record's hash is missing or `null`). A calibrated policy that records a hash accepts only that fit, always, independent of the opt-in fingerprint check: `DecisionGate` raises `PolicyIntegrityError` for another `fits.json`, and `s1 drift` and both exporters refuse a record whose `calib.fit_sha256` differs, is `null`, or is missing (run `s1 apply` again with the policy's `fits.json`). A policy with `fit_sha256: null` takes any fit.
- **Strict Inputs**: records with NaN or infinite values, JSON booleans as numbers, or probabilities that do not sum to `1 ± 0.02` are refused with their file and line number; score keys must be 1–20 ASCII digits naming distinct levels up to `2**53` (`"0"` and `"00"` together are refused), and wherever the calibrated distribution is read (`s1 drift`, both exporters, `--use-calib`) a calibrated choice or score distribution must have exactly the raw keys. `s1 gate --p` / `evaluate_noul(p_raw=...)` takes a finite number in `[0, 1]` (not a boolean), and `--probs` / `probs_raw` at least two classes summing to `1 ± 0.02`, renormalized before routing. `policy.json` and `fits.json` hold one entry per group; an `escalation_band` needs `0 <= band_lo <= band_hi <= 1`. Malformed input raises `ValueError` in the SDK and exits `1` in the CLI.
- **Reserve-Then-Settle Budget Ledger (`s1 ledger` & `s1 probe`)**: Uses exclusive `flock` on `<journal>.lock`, rejects hard-linked ledgers, and halts immediately if a measured cost exceeds its reservation (`LedgerUnusable`).

---

## CLI Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Usage error, missing or malformed input, no matching groups, `lint check --strict` findings, policy fingerprint mismatch, ledger error, `s1 demo` target not new or empty |
| `2` | `s1 apply`, or `s1 drift` or `s1 export` with `--fit`, refused an in-sample item (override with `--allow-in-sample`) |
| `3` | `model_served` mismatch against the fit or policy (`s1 apply` without `--pass-through-unfitted`, `s1 gate`) |
| `4` | `s1 drift --fail-on-critical` found a `CRITICAL` question |

Errors print a single `s1: <message>` line on stderr; non-printable characters in it (a newline or an escape sequence inside an id or a path) are written escaped, as `\n` or `\x1b`, while printable text, non-ASCII included, is kept. When an input that another command produces is missing, the message names that command (for example `run \`s1 fit\` first`). Set `S1_DEBUG=1` to re-raise errors raised by the library, with their traceback; checks the CLI makes itself (a missing input file, an empty grades file) still print one line. A missing input path is an error everywhere, including `s1 ledger status`, except `s1 lint check`, which reports an unreadable questions file as a `JL000` finding in its normal (`--format json`) report.

---

## Development

From a source checkout (the sdist includes the test suite), all tests run without API keys or calls to external services (a few backend tests talk to an HTTP server on `127.0.0.1`):

```bash
uv sync --all-extras
uv run pytest -v
```
