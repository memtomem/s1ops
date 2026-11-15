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

LLMOps platforms such as **Langfuse**, **Arize Phoenix** and **LangSmith** are built around **System 2 generative workflows** (free-text completions, prompt templates, multi-turn RAG chains, token streaming latency). For **System 1 decision models**, `s1ops` covers decision-model operations across seven dimensions:

| Dimension | Text-oriented LLMOps tooling (typical focus) | `s1ops` (System 1 LLMOps) |
| --- | --- | --- |
| **Primary Output** | Unstructured text (`string`) | Calibrated probability vector (`noul`: `p` in `[0, 1]`; `choice`: a distribution over K options; `score`: ordinal `0..L-1`) |
| **Aggregation Unit** | Trace, session or prompt version | Strict **`(question_id, model_served, primitive)`** group — never pools across wordings or models (drift's per-question report pools models on purpose, so a model change shows) |
| **Calibration & Small-n Bias** | Scores attached to traces; calibration fitting and small-sample noise floors are outside the usual workflow | **Equal-width/mass ECE**, **Monte Carlo Calibrated Noise Floor (`p95`)**, **Tie-Mass rounding detector**, **Temp/Platt/Isotonic** post-hoc fit |
| **Decision Thresholds & Routing** | Left to application code, often a fixed cutoff | **Tie-safe Error-Budget Threshold (`select_threshold` + Clopper-Pearson)**, **System 1 → System 2 Escalation Band (`[lo, hi)`)**, & **`DecisionGate` Runtime Evaluator** |
| **Prompt / Question QA** | Prompt playgrounds and versioning | **Static Linter (`JL000`–`JL060`)** + **Probe Coherence** (Noul vs Yes/No Choice twin gap & inverse sum `abs(p + p_inv - 1)`) |
| **Distribution & Model Drift** | Latency, token, cost and embedding-level monitoring | **Label-Free Probability PSI & Jensen-Shannon Divergence**, **Silent `model_served` Upgrade Alert**, & **Escalation Band Rate Shift** |
| **Data Privacy Default** | Prompt and response inspection for debugging | **`safe-v1` Zero-State Default**: never writes raw item `state` into records, probe ledgers, or exported spans/scores |

`s1ops` bridges both worlds: you run rigorous System 1 calibration, linting, drift monitoring, and policy compilation locally (`s1` CLI + `s1 web`), execute sub-millisecond runtime gating via `s1.DecisionGate`, and export **`safe-v1` body-free scores, annotation queues, and OpenInference spans** as local files for **Langfuse** and **Arize Phoenix**; `s1 export` uploads nothing.

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

**Outputs**: compiled `fits.json` and `policy.json`, a blind review sheet, `safe-v1` Langfuse runs/scores/queues, Phoenix / OpenInference `GUARDRAIL` / `EVALUATOR` / `RERANKER` spans (noul / choice / score), and a local web workbench on `127.0.0.1:8765`. The state store is read by two commands only: `s1 labelaudit sheet` writes the state text of every item under review (the top-ranked flags plus random controls) into the review sheet, which is meant for a human reviewer (`s1 web --sheet` shows it in the browser, to this computer only), and `s1 probe` sends each state to the selected backend (a live API for `--backend systemone` or `openai`). s1ops never writes that state text into records, ledgers, probe output or exports; `s1 apply`, which copies its input rows, leaves a top-level `state` out.

---

## Quickstart

### 1. Installation

```bash
pip install s1ops                # or: uv add s1ops
pip install "s1ops[langfuse]"    # optional: the Langfuse client, for your own upload of `s1 export` files
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
- `states_local.jsonl`: Synthetic item states for the blind review sheet and the probe runner. One row per item, `{"dataset", "item_id", "state"}`; both commands check `item_id`, `state`, `dataset` and `split` the same way (`s1 probe` also checks a `label`) and read a row without `dataset` as `dataset: "default"`, the value `s1 probe` writes into its records. To match records that carry no `dataset`, write `"dataset": null`. `s1 probe` refuses a states file with no rows (empty, blank, or `[]`) before it sends anything or opens its output.
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
# Route Noul groups by the error-budget threshold, so the budget bounds the accepted answers' error
s1 policy tune s1-demo/records.jsonl -o s1-demo/policy_threshold.json --error-budget 0.10 --noul-routing threshold --use-calib
s1 policy curve s1-demo/records.jsonl --use-calib

# 3b) The same for choice and score answers
s1 fit s1-demo/records_categorical_train.jsonl -o s1-demo/fits_categorical.json --method temperature
s1 apply s1-demo/records_categorical.jsonl --fit s1-demo/fits_categorical.json -o s1-demo/records_categorical_calibrated.jsonl
s1 audit s1-demo/records_categorical.jsonl --fit s1-demo/fits_categorical.json
s1 policy tune s1-demo/records_categorical.jsonl -o s1-demo/policy_categorical.json --error-budget 0.10 --use-calib

# 4) Answerability Pre-Flight & Benjamini-Hochberg FDR Error Slice Mining
s1 slice answerable s1-demo/records.jsonl --features s1-demo/features.jsonl
s1 slice mine s1-demo/records.jsonl --features s1-demo/features.jsonl --fdr-q 0.10
s1 slice answerable s1-demo/records.jsonl --features s1-demo/features.jsonl --use-calib

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
#    --raw-only, the 0.1.0 name of the default, cannot be combined with it; the probabilities in a stored `calib` object are validated either way)
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

**After the review.** The reviewer can answer in the workbench instead of an editor (`s1 web --sheet SHEET`, the Blind Sheet Review view) and download the completed sheet. When the reviewer has written an answer under every item, `s1 labelaudit estimate SHEET --key KEY` scores the sheet against the key, and `s1 labelaudit patch RECORDS --sheet SHEET --key KEY -o RELABELED.jsonl` applies it: where the reviewer answered otherwise than the label, every labeled record of that item and question takes the answer, whichever model answered it and a failed probe's row as well, as a noul boolean, a choice option or a score level. Records of other questions, unlabeled records and items that were not on the sheet are copied as they are, without a top-level `state`. The sheet, the key and the records must be of one review. A sheet that asks another question than the key's (its `Question ID:` block), or answers an item the key does not have, exits 1, as does a key that lists one item twice; so does a reviewed item that is missing from the records or carries a label the review was not of. Nothing is written then. Two sheets of one question cannot be told apart this way, so keep each sheet with its own key. A second run over the patched file changes nothing more (`records_already`). The summary lists each correction by id with its old and new label. One reviewer's answer becomes the label, so decide from `estimate` whether that reviewer should; flagged items are the ones a model disagreed with, so a patch moves labels toward the models, which the control arm is there to measure. Fits and policies made from the old labels are not changed and no longer describe the file: run `s1 fit` and `s1 policy tune` again. Labels in a states file are not changed either, so a new `s1 probe` from it writes the old ones.

**Reviewing in Langfuse.** A review can also be done in a Langfuse annotation queue over the observations of `s1 export langfuse`. Give the queue a score config named `s1.<question_id>.label` for each question, the reviewer's answer: BOOLEAN for a noul question, CATEGORICAL over the options for a choice question, and NUMERIC (or CATEGORICAL) over the levels for a score question. s1ops makes no network call, so fetch the scores yourself, e.g. `GET /api/public/v3/scores?name=s1.<question_id>.label&fields=subject` (join the pages into one JSON list, or keep one response), and then run `s1 labelaudit pull RECORDS --export EXPORT.json --scores SCORES.json -o RELABELED.jsonl`. A score on an exported observation, or on its trace (one trace is one item), reviews that item and question, and the answer is applied as `patch` applies a sheet's, to the same records and with the same refusals, taking the item's label in the records as the one the review was of. Scores of other names are not read, and label scores on observations and traces the export does not hold are left out and counted (`scores_elsewhere`); a file with none for this export exits 1. Two scores of one item and question that answer differently exit 1, as does an answer the question cannot take (a NUMERIC on a noul question, an option it does not have), a score without its `subject` and a label score on a session. The Langfuse queue shows the model's answer next to the observation, so this review is not blind: there is no control arm, and `estimate` does not apply to it.

**Live probes.** `s1 probe --backend systemone` sends each state and question to `POST /v1/systemone` (key in `TYPESAFE_API_KEY`, endpoint in `TYPESAFE_BASE_URL` or `--base-url`); `--backend openai` sends them to OpenAI's `POST /v1/decisions` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`) with prompt caching disabled. Both need `--model`; only the offline mock has a default. A response that reports a cache write has no known price and is settled at its full reservation; cached reads, which are discounted, are charged at the full input price, an upper bound. A question that API cannot take as written — a Noul with `criteria`, object (rather than string or input-part list) instructions, or option and level counts outside its limits — is refused before anything is reserved, rather than sent in a different form under the same `question_id`. A refused answer is recorded with `error_code: "refusal"`. `/v1/systemone` resolves an alias to the version that answered (`jev-latest` came back as `jev-1.14.0`), so a silent upgrade shows in `model_served`; `/v1/decisions` returned the alias `gpt-6-luna` itself, so a model change behind it is visible only as distribution drift. `model_served` is whatever the server reports: a compatible server that echoes the request records the name you passed, and one that names every checkpoint alike leaves a checkpoint change visible only as distribution drift too, since drift's model comparison and the exit 3 of `s1 apply` and `s1 gate` compare names. The APIs report billable input tokens, not dollars, so `--usd-per-input-token` is required. Every attempt reserves its worst case first (`--worst-usd-per-token`, plus `--reserve-overhead-tokens` for the prompt tokens a server bills on top of the state and question: 300 for systemone and 200 for openai by default: a short state was billed up to 216 and 98 tokens more than estimated, and about 236 and 118 extrapolated to an empty one) and is settled as measured, or, when the request may have been billed without a usable reported cost (timeout, connection error, HTTP error), at its full reservation. Timeouts, connection errors, 408, 429 and 5xx are retried up to `--max-retries` times. An item that still fails is written as a record with `model_served: null` and an `error_code`, which `s1 drift` counts as a failure. Records are written as they arrive, so a run stopped by `--cap-usd` or by a measured cost above its reservation keeps what it paid for. Error messages carry only an error code, never server text, since a server may echo the request.

**Probe coherence.** Unless `--no-expand-probes` is given, `s1 probe` sends each Noul with string instructions, or with a supplied inverse (below), three ways: as written (`probe_role: "noul"`), as a yes/no Choice with the same instructions (`twin`), and as its inverse (`inverse`). `s1 lint probe-audit` then reports, per question and served model, the twin gap `|p - p_yes|` and the inverse error `|p + p_inv - 1|`. Write the inverse yourself when you can, under the question's lint extension:

```json
{"positive": {"type": "noul", "instructions": "Is this movie review positive?",
              "criteria": {"true": "positive", "false": "negative"},
              "x-s1-lint": {"inverse": {"instructions": "Is this movie review negative?",
                                        "criteria": {"true": "negative", "false": "positive"}}}}}
```

The extension is stripped before hashing, so the question keeps the `question_id` `s1 lint check` gives it. Without one, the inverse is generated as `"Is it false that: <instructions>"` with the true/false criteria swapped; a model that reads that sentence literally answers it like the original, so its error can measure the generated wording more than the model (on a local Kev, s1ops 0.6.0's generated inverse, which also dropped the criteria, gave an inverse flip rate of 1.0 next to a twin gap of 0.012). A Noul with object or list instructions and no supplied inverse is sent once, as written, and the audit skips it. Each inverse record says which wording it got (`probe_inverse`: `supplied` or `generated`), and the audit reports it as `inverse_source` (`null` for records from before the tag, or a mix). A question and model whose inverse records carry more than one `question_id` are refused rather than pooled, so audit the output of one wording at a time. A question carries `x-s1-lint` or `x-jev-lint`, not both, and `s1 probe` refuses a malformed extension, as `s1 lint check` reports it, before the ledger or the output is opened.

### 4. Launch the Local Web Workbench

```bash
s1 web --workspace s1-demo --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. The workbench opens on an **Overview** of the workspace: a status table of the views (Compare and its judge audit on two rows; the Simulator, a what-if, has no status), records, groups (with whether each has a fit), and each workspace file, present or missing, with the command that writes it. The status table requests nothing until you press *Check* on a row or *Check all*; each row then counts its view's items (groups, findings, questions, checkpoints, the judge audit) once each, at the worst status badge that view shows for it (Attention, Warnings, Other, OK), and says when nothing was there to count. A row not checked yet, or without data, is not OK. Calibration is not audited from the Overview past six labeled groups, as its own view lists them instead. A group filter applies to the Calibration, Simulator and Slices views (the label-audit ranking then covers that group's records only); Compare and Drift have a question picker of their own, since their Holm-adjusted p-values and summary are computed over every question, and Checkpoints a checkpoint picker. The view, the group, the question and the simulator's inputs are kept in the URL, so a link reopens the same view and Back restores the previous one. Light, dark or the system's theme. Its views:
1. **Calibration & Noise Floor**: reliability diagram before and after the fit, each bin with its 95% Wilson interval (counted from the bin's labels) and its record count, in equal-mass or equal-width bins; the metrics with their bootstrap 95% intervals (the number of resamples is a control); the audit's warnings; the Monte Carlo 95th-percentile ECE noise floor and the tie-mass rounding detector. With no group chosen and more than six labeled groups, it lists the groups instead of bootstrapping every one. Its tables sort, filter and export the rows shown as CSV (a text cell starting with `=`, `+`, `-`, `@`, a tab or a carriage return gets a leading `'`) or JSON (values as served).
2. **Threshold & Escalation Simulator**: Interactive `band_lo`, `band_hi`, `cost_s1`, `cost_sys2`, and `error_budget` controls, on raw or calibrated probabilities (a Probabilities control; Auto picks calibrated when the group's records carry `calib`). For every primitive: the threshold operating curve (error rate among accepted answers against coverage, shaded up to its Clopper-Pearson 95% upper bound, with the error budget and the tuned threshold marked) and the cost per 1,000 decisions with its System 2 and error parts; the caption states the error cost and the System 2 accuracy that cost assumes. For a noul group also: the live System 1 Auto vs System 2 Escalation split, the active band's confusion table and cost, the recommended band, and the 0.10 band sweep with its Pareto frontier as a chart and a table. The curve and band tables sort, filter and export as CSV or JSON.
3. **Question Lint & Probe Coherence**: Static `JL000`–`JL060` findings with their severity counts (suppressed ones counted), each finding's rule title, fix and evidence, and the rule catalog; a workspace without a questions file says so instead of showing 0 findings. Per probe group, the Noul/Choice twin gap and polarity inverse sum statistics, a scatter of each item's twin answer and of its inverse answer against the question as asked (with the line coherent answers lie on), and the items as a table (a group's first 2,000 items; `s1 lint probe-audit` prints them all). The tables sort, filter and export as CSV or JSON.
4. **Slices & Blind Label Audit**: per group, a forest of each slice's accuracy against its null baseline (both with their Wilson 95% intervals) and a dumbbell of each tested feature's error rate inside and outside it, with Benjamini-Hochberg FDR error slices at a chosen `q` (0.01 to 0.20, default 0.10; the p- and q-values are in the table), both predicting from raw probabilities (`/api/slices?use_calib=true` for the calibrated ones). Below them, the top label disagreement candidates (15 to 100), with how many records were ranked and left out, and whether the margins are calibrated (a fit file was read) or raw. The tables sort, filter and export as CSV or JSON.
5. **Shadow Compare & Judge Audit**: A dumbbell of each question's primary and shadow accuracy (`jev-1.13.0` vs `kev-1.0` in the demo) over its labeled pairs, a table of every compared question (flip rate, distances, exact McNemar p-values raw and Holm-adjusted, whether the shadow is replaceable and under which guard), and per question its flipped pairs (the report lists up to 50). With more than six questions the view lists them and opens the one you choose. The judge audit leads with the agreement over every graded item, then per-criterion quadratic-weighted κ, bias and pass agreement against the stated bar. The tables sort, filter and export as CSV or JSON.
6. **Distribution & Model Drift**: Baseline vs Canary PSI with its permutation p-value and Jensen-Shannon Divergence on raw probabilities, mean/entropy/escalation rate shift, and silent `model_served` upgrade alerts (`CRITICAL` / `WARN` / `OK`); a table of every question, most severe first, and per question the two score distributions overlaid in the PSI's bins, their quantiles, and every per-side statistic next to each other.
7. **Checkpoints & Training Runs** (read-only, `GET /api/checkpoints`): every name the workspace holds, as the `checkpoint` of a training manifest (`training_manifest*.json`) or of a training-run record (`trainrun*.json`) in the workspace folder, or as the `model_served` of a record in `records.jsonl`, `records_kev.jsonl` or `records_canary.jsonl`. Per name: the size of its training set (inherited items included), its answered records per file and question with how many of their items it trained on, counted by `--training-manifest`'s rule (the workbench's other views read the records without a manifest, so this is where a trained item shows), the run record as it is on disk (trainer settings, summary metrics, weights digests) with one chart per step metric, and the policies for that model with whether each names the recorded run. It says when a run was recorded against another manifest file than the one here, when its parent's record is missing, and when a checkpoint answered no record; a model without a manifest has an unknown training set, never a count of zero. Two different run records for one checkpoint, or a malformed file under either name, is a 400 naming the file. Nothing is written and no training is started. See [Fine-Tuning](#fine-tuning-around-an-external-trainer).
8. **Blind Sheet Review** (`s1 web --sheet s1-demo/blind_sheet.md`, `GET /api/sheet`): answer a sheet from `s1 labelaudit sheet` one item at a time, by mouse or keys (`j` / `k` move, `1`–`9` answer, `u` clears, `n` finds the next unanswered item), then download the completed sheet for `s1 labelaudit estimate` and `patch`. The answers stay in this browser until then (per sheet file: editing the file starts a new draft). The sheet holds state text, so it is served to this computer only: `--sheet` needs `--host 127.0.0.1` or `localhost`, and the route refuses any request that is not from a loopback address, names another Host, or carries a proxy header. It is never looked for in the workspace, and the view requests nothing else, so the key's labels and the ranking stay out of sight. The web reads a sheet as `s1 labelaudit sheet` writes it: an item edited by hand (an answer on the next line, a note after it) is refused, naming the item, and `estimate` still scores it. A sheet asks options joined with `, `, so a choice question with an option holding `, ` cannot be put on one (`patch` and `pull` still take it).

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

- **Zero-State Privacy (`safe-v1`)**: record files, the ledgers `s1 probe` writes, probe outputs, and Langfuse/Phoenix exports never store raw item `state` text. A record's `state_sha256`, which both exporters copy out, is refused when it is read unless it is absent, `null`, or the 64 lowercase hex characters of a SHA-256 digest, so a records file cannot carry text to an export under that name; the error says the value's type and length and does not show it. `s1 apply` copies its input rows, so it leaves a top-level `state` out of every row it writes (`state_sha256` and the other fields are kept) and adds `"state_dropped": N` to its JSON summary when any input record had one. `Ledger.reserve` / `settle` refuse a `state` metadata key (`ValueError`, before anything is written), and `s1 ledger recover` refuses a journal whose entries carry one rather than copy it into a new file; a journal written that way by s1ops 0.2.0 or earlier still opens.
- **One Routing Rule**: `DecisionGate`, `s1 drift`'s escalation rate, and the `policy_action` of both exporters apply a policy through the same rule. Every one of them routes the distribution the policy was tuned on: the calibrated one exactly when the policy was compiled with `--use-calib` (`calibrated: true`), whatever a record carries. The reports take it from the record's `calib`, so a calibrated policy needs `s1 apply` output, or `--fit` on `s1 drift` and `s1 export`, and refuses a record without `calib`; `s1 drift` applies to each record only the policy of its own `(question_id, model_served, primitive)` group, so records from another model (the demo canary's `jev-1.14.0-rc1` third) are left out of the escalation rate, which is `null` when no record has a policy. The distributions `s1 drift` compares (PSI, JS divergence, quantiles, entropy) are a separate choice: raw on both sides, or calibrated on both with `--calibrated`, which refuses an answered record without `calib` rather than compare it raw, and records of one `(question_id, model_served, primitive)` group calibrated through different fits (`calib.fit_sha256`), since a refit would show as drift; the report says which in `distribution`. s1ops 0.1.0 and earlier compared each record's `calib` when present and its raw probabilities otherwise, and their drift reports have no `distribution`. A Noul `error_budget` policy accepts when `max(p, 1 - p)` reaches the threshold (else escalates); a band accepts at or above `band_hi`, blocks below `band_lo`, and escalates between. `s1 policy tune --max-false-pass RATE` writes a Noul band with `band_lo == band_hi`, which never escalates: its cutoff is the smallest observed probability (or `1.0`) at which at most `RATE` of the group's label-false items reach it (with `--use-cp-upper`, its Clopper-Pearson upper bound), and `metrics_on_split` records the counts it was chosen on (every other policy records what it does on the split: `coverage`, `accepted`, `errors`, `accepted_error`); it replaces the F1 band, so `--max-escalation` is refused with it, and choice and score groups keep the error budget. A choice or score policy selects the top answer when its probability reaches the threshold, and escalates on a tie at the top or when no threshold met the error budget (`threshold: null`); since a tie escalates at any threshold, `s1 policy tune`, `s1 policy curve` and the web simulator leave it out of every accepted error and count it only among the items (coverage, delegation). Retuning choice or score data that has such ties can therefore change the threshold, and with it `policy_sha256`, from what s1ops 0.8.0 and earlier wrote; a policy already compiled routes as before; a policy tuned with `--use-calib` must be gated with its `fits.json`. Exports of policy-routed records carry `routing_rule` (`s1.routing_rule` on Phoenix spans, `metadata.routing_rule` on Langfuse observations), currently `"3"`; exports from s1ops 0.1.0 and earlier have none, and their `policy_action` and `predicted` may differ for the same policy. Choice and score probabilities are renormalized with an exactly rounded sum, so retune choice or score policies compiled by 0.1.0 or earlier on Python 3.11 (run `s1 apply` first for `--use-calib` ones).
- **A Noul Band Is Not Bound by the Budget**: every policy records in `metrics_on_split` what it does on the split it was tuned on, by the routing rule itself. For a Noul group `s1 policy tune` picks an escalation band that maximizes F1 under `--max-escalation`, and the band then routes; `--error-budget` and `--use-cp-upper` govern the separate `threshold`, not the band. So `accepted_error` can exceed the budget: on the demo records, raw, 0.238 at a budget of 0.10. `--max-escalation` does not help: at 0 the band is empty, one cutoff that sends nothing to System 2 and accepts every item (0.275 there), not the threshold. `--noul-routing threshold` routes a Noul group by the threshold instead (`mode: error_budget`), so the budget bounds `accepted_error` on the split: 74 accepted, 7 wrong (0.095) there, with the other 86 escalated; it counts an item at `p == 0.5` as the rule answers it, true, and refuses `--max-escalation` and `--max-false-pass`. `--max-false-pass` replaces the band with a cutoff that bounds the label-false items that pass.
- **No Cross-Group Pooling**: Metrics, fits, thresholds, and slice reports are keyed strictly to `(question_id, model_served, primitive)`; `s1 slice answerable` / `mine` and the workbench's Slices tab report each group separately. Two commands are exceptions. Drift reports per `question_id` and pools its models, so a changed `model_served` shows (CRITICAL); its policy lookup and calibrated fit check stay per group, and it refuses a question whose records name more than one primitive (a `question_id` names one wire question). `s1 compare` refuses a question whose paired records on one side come from more than one model or primitive; split such a file by `model_served`.
- **Drift Alerts Need Evidence (`s1 drift`)**: PSI of two samples from one distribution grows with smaller samples (about 0.33 at 54 records per side, over the `CRITICAL` threshold of 0.25), so a PSI over a threshold raises the status only when a permutation test puts it outside sampling noise: `CRITICAL` needs PSI >= 0.25 and p < 0.01, `WARN` needs PSI >= 0.10 and p < 0.05. The p-value (`psi_p_value`) is a Monte Carlo estimate from 1,000 resamples of the answered records' bin counts (seeded per question; the draws can differ under another numpy, which matters only for a p near alpha). A PSI over a threshold that is not significant is listed in `notes`, not `alerts`; each question also reports `n_ref_answered` / `n_cur_answered`, the records the test used. `OK` does not establish stability: small samples miss real shifts (in one measurement, 50 of 120 comparisons with a moderate shift at about 54 records per side stayed `OK`). The test treats records as exchangeable, so it does not fit autocorrelated windows, and its alpha holds per question and per comparison, not across questions or repeated runs. Model changes, `model_requested` mismatches and failure-rate jumps alert as before. The latency summary is informational: a statistic is `null` when no record has a numeric `latency_ms`, or when it is not finite (`latency_ms` is not validated, so very large values can overflow it). `audit_drift(..., psi_resamples=0)` restores the threshold-only rule. s1ops 0.6.0 and earlier alerted on PSI alone.
- **Out-of-Sample & Model-Mismatch Enforcement (`s1 apply` & `DecisionGate`)**:
  - Attempting to apply a calibration fit to any `(dataset, item_id)` used during `s1 fit` exits with **code `2`**.
  - With `--training-manifest`, fitting or evaluating a checkpoint on an item it trained on also exits with **code `2`** ([Fine-Tuning](#fine-tuning-around-an-external-trainer)).
  - Attempting to apply a calibration fit or runtime `DecisionGate` to a record whose `model_served` differs from the fitted/compiled model exits with **code `3`** (`ModelServedMismatchError`). For a production file that mixes models, `s1 apply --pass-through-unfitted` calibrates the records it has a fit for, writes the others unchanged (except a top-level `state`, as above), and prints how many of each (with the reason per group); class mismatches and in-sample items are still refused.
  - `s1 drift --fit` and `s1 export langfuse|phoenix --fit` do the same in memory, so no intermediate file is needed: reports and export files equal the ones made through `s1 apply --pass-through-unfitted`, a stored `calib` is replaced on a record the fit file has a fit for and kept on any other, in-sample items exit `2` (`--allow-in-sample`), and class mismatches exit `1`. A refusal to apply the fit, and an input line the record reader refuses, name the input file and its own line. `s1 drift`'s report and the printed export summaries gain a `fit` block with what was calibrated and what was passed through, per file; an export file is unchanged. `s1 drift --fit` needs `--policy` or `--calibrated`, since a raw comparison without a policy compares and routes no calibrated probabilities. The web workbench does not take a fit. In the SDK: `apply_fits(read_records(path), fits, pass_through_unfitted=True).records`.
- **Policy Fingerprint (`policy_sha256`)**: every `PolicyEntry` carries a canonical SHA-256 fingerprint of its routing fields. Verification is opt-in: pass the expected fingerprint to `DecisionGate.evaluate_noul(expected_policy_sha256=...)` or `s1 gate --expected-sha256`, and a mismatch raises `PolicyIntegrityError`. Loading `policy.json` also compares each row's stored fingerprint with one recomputed from its fields and refuses a mismatch (`PolicyFingerprintError`, CLI exit 1, web 400), so a threshold or band edited after `s1 policy tune` wrote the file is caught at load. To accept a deliberate hand edit, remove that row's `policy_sha256`; a row without one is only recomputed.
- **Fit Provenance (`fit_sha256`)**: `s1 apply` writes the fit's `fit_sha256` into each record's `calib`. `s1 policy tune --use-calib` refuses a selected record without calibrated probabilities, a non-null `calib.fit_sha256` that is not a string, and a group whose records were calibrated through different fits (exit `1`, naming the records file), and records the group's hash in the policy (`null` when every record's hash is missing or `null`). `s1 policy curve --use-calib` refuses the same records in the groups it selects, and `s1 slice answerable` / `mine --use-calib` and `/api/slices?use_calib=true` in the groups they report (exit `1` / HTTP 400, naming the records file), and so does the web workbench's policy simulator (`/api/simulate-policy`) for the group it simulates on calibrated probabilities (HTTP 400 naming the records file). Without `use_calib` the simulator picks: calibrated when any labeled, answered record of the group carries `calib` (so a group mixing calibrated and raw records is still refused), raw when none does; `use_calib=false` or `true` chooses, and the response's `distribution` says which was used. s1ops 0.9.0 and 0.10.0 defaulted to calibrated and refused a group without `calib` unless `use_calib=false` was sent; earlier versions simulated a record without `calib` on its raw probabilities as if they were calibrated. A calibrated policy that records a hash accepts only that fit, always, independent of the opt-in fingerprint check: `DecisionGate` raises `PolicyIntegrityError` for another `fits.json`, and `s1 drift` and both exporters refuse a record whose `calib.fit_sha256` differs, is `null`, or is missing (run `s1 apply` again with the policy's `fits.json`). A policy with `fit_sha256: null` takes any fit.
- **Strict Inputs**: records with NaN or infinite values (in a records file, `NaN`, `Infinity` or a literal that overflows to infinity, such as `1e400`, anywhere in a row, whether anything reads that field or not; a row already decoded and passed to `parse_record` has only the fields it reads checked), a lone surrogate (half of a UTF-16 pair, which JSON can escape on its own and no UTF-8 file can hold; refused under the same two conditions, so no command writes part of a `fits.json` or `policy.json` and stops at one), JSON booleans as numbers, or probabilities that do not sum to `1 ± 0.02` are refused with their file and line number; score keys must be 1–20 ASCII digits naming distinct levels up to `2**53` (`"0"` and `"00"` together are refused), and wherever the calibrated distribution is read (`s1 drift`, both exporters, `--use-calib`) a calibrated choice or score distribution must have exactly the raw keys. `s1 probe` and `run_batch` hold a states row's `label` to the record rule of every question it goes onto (one `label` goes onto each question's record, so a row cannot label a noul and a choice in one file), and refuse it naming the row before anything is sent; a lone surrogate in a states row's `item_id`, `dataset` or `split`, in the model name, or in a question's name or wording is refused there too, since those strings are copied into the records (a question's name into its probe records) and a wording with one cannot be given a `question_id`. A custom `Backend`'s `answer` contributes only the answer keys (`p`; `choice`, `probabilities`, `confidence`; `score`, `probabilities`, `confidence`), and a response that would not read back as a record is recorded as `invalid_response`, once its cost is settled, so a run leaves only records the reader takes (a `cost_usd` the ledger refuses stops the run, with no record of that call). `s1 gate --p` / `evaluate_noul(p_raw=...)` takes a finite number in `[0, 1]` (not a boolean), and `--probs` / `probs_raw` at least two classes summing to `1 ± 0.02`, renormalized before routing. `policy.json` and `fits.json` hold one entry per group; an `escalation_band` needs `0 <= band_lo <= band_hi <= 1`. Malformed input raises `ValueError` in the SDK and exits `1` in the CLI.
- **Reserve-Then-Settle Budget Ledger (`s1 ledger` & `s1 probe`)**: Uses exclusive `flock` on `<journal>.lock`, rejects hard-linked ledgers, and halts immediately if a measured cost exceeds its reservation (`LedgerUnusable`).

---

## Fine-Tuning Around an External Trainer

s1ops starts no training. It writes down what a checkpoint will train on and refuses to fit or evaluate that checkpoint on those items. The trainer is yours; [Kev](https://github.com/jaredpalmer/kev)'s `kev.train --data` takes the rows the recipe below writes.

```bash
# These run as written on the offline mock, which answers the same whatever the name, so its results judge nothing. Against a
# real server, add to every probe: --backend systemone --base-url URL --usd-per-input-token RATE (see Live probes).
#
# questions.json holds the one question these labels answer, here {"tickets": {"type": "noul", "instructions": "..."}}. One
# `label` goes onto every question in the file, so `s1 probe` refuses a label that does not suit them all, which the demo's
# questions.json, mixing noul, choice and score, would be.

# 1. Give each states row a `split` (train, cal, eval); `s1 probe` copies it into its records, as it does `label`:
#    {"dataset": "tickets", "item_id": "t1", "state": "...", "split": "train", "label": true}
s1 probe questions.json --no-expand-probes --model current-1 --states states_local.jsonl -o current.jsonl --ledger ledger.jsonl

# 2. Name the new checkpoint and write what it trains on. `--parent previous_manifest.json` adds what a checkpoint it
#    starts from trained on.
s1 trainset export current.jsonl --split train --checkpoint tickets-r1 -o training_manifest.json

# 3. Build train.jsonl with the recipe below, train outside s1ops, then serve the checkpoint under that name and probe the
#    held-out items, with the cal and eval rows of the states file in files of their own. Probe the current model on the
#    same eval rows for step 4's drift.
python3 -c 'import json, sys; sys.stdout.writelines(r for r in sys.stdin if r.strip() and json.loads(r).get("split") == sys.argv[1])' cal < states_local.jsonl > states_cal.jsonl
python3 -c 'import json, sys; sys.stdout.writelines(r for r in sys.stdin if r.strip() and json.loads(r).get("split") == sys.argv[1])' eval < states_local.jsonl > states_eval.jsonl
s1 probe questions.json --no-expand-probes --model current-1 --states states_eval.jsonl -o current_eval.jsonl --ledger ledger.jsonl
s1 probe questions.json --no-expand-probes --model tickets-r1 --states states_cal.jsonl -o candidate_cal.jsonl --ledger ledger.jsonl
s1 probe questions.json --no-expand-probes --model tickets-r1 --states states_eval.jsonl -o candidate_eval.jsonl --ledger ledger.jsonl

# 3b. Record the finished run next to its manifest: exit 2 if train.jsonl holds an item the manifest does not list.
s1 trainset record training_manifest.json --data train.jsonl --weights out/adapter_model.safetensors -o trainrun.json

# 4. Promotion check. A new name starts with no fit and no policy: the current model's `s1 apply` exits 3 on it, and `DecisionGate`
#    raises `ModelServedMismatchError`.
#    `--training-run` adds the run's `trainrun_id` to what `compare` and `policy` print, and the record to a Langfuse export.
s1 compare current.jsonl candidate_eval.jsonl --training-manifest training_manifest.json --training-run trainrun.json
s1 fit candidate_cal.jsonl --training-manifest training_manifest.json -o fits.json
s1 apply candidate_eval.jsonl --fit fits.json -o candidate_eval_calibrated.jsonl
s1 policy tune candidate_eval_calibrated.jsonl --use-calib --training-manifest training_manifest.json --training-run trainrun.json -o policy.json
s1 drift current_eval.jsonl candidate_eval.jsonl
s1 export langfuse candidate_eval_calibrated.jsonl --policy policy.json --training-run trainrun.json -o langfuse_export.json
```

`s1 promote` runs step 4 as one command, with the manifest on every step that takes one, so no step can be run without it:

```bash
s1 promote --current current_eval.jsonl --cal candidate_cal.jsonl --eval candidate_eval.jsonl \
    --training-manifest training_manifest.json --training-run trainrun.json -o promotion/
```

It takes records already probed and calls no backend. It writes `fits.json`, `candidate_eval_calibrated.jsonl` and `policy.json` into the directory, the files the commands above write, and prints one `s1-promotion/1` report, also saved as `promotion.json`: each step's own report under `compare`, `drift`, `fit`, `apply` and `policy`, the candidate's name, and `replaceable`, true when `s1 compare` found every question replaceable. `--current` is the model in place answering the candidate's evaluation items; `--cal` and `--eval` are each answered by the candidate alone, one `--training-manifest` must be for it, and every question of `--eval` must share an item with `--current`: a question the model in place never answered would get a policy with nothing compared. A manifest for the model in place goes to `s1 compare` only. `--method`, `--min-n` and the options of `s1 policy tune` (`--error-budget`, `--noul-routing`, `--max-false-pass`, ...) mean what they mean there. A step that refuses keeps its exit code and names itself (`s1: promote: fit: ...`, exit 2 for a trained item). `compare` and `drift` run before anything is written; a later step that fails leaves the files of the steps before it. Each input is read by more than one step, so a FIFO is refused. It exits 0 whatever `replaceable` says: read the report.

`s1 drift` across a rename reports every question `CRITICAL` with the alert that `model_served` changed, and `--fail-on-critical` exits 4, so read its PSI and quantiles here rather than its status. A later run against the promoted checkpoint's own records is the monitor. A probe's states row holds one `label`, and it goes onto every question in the questions file, so when the labels differ give each question its own questions file and states file.

**The manifest** (`s1-training-manifest/1`) holds `(dataset, item_id)`, `question_id` and label for each row, plus the checkpoint's name, the split and the records file's SHA-256. It holds no `state` text and no question wording, so `s1 probe` and `s1 labelaudit sheet` stay the only readers of a states file. The trainer joins them from its own local files, checking the wording by `question_id`:

```python
import json

from s1ops import question_id
from s1ops.lint.loader import read_question_entries
from s1ops.monitor.runner import probe_tasks

manifest = json.load(open("training_manifest.json", encoding="utf-8"))
entries, errors = read_question_entries("questions.json")  # read as `s1 probe` reads the file
if errors:
    raise SystemExit("; ".join(errors))
tasks = probe_tasks(entries, probe_mode=False)
wording = {question_id(t.question): t.question.wire_dict() for t in tasks}
states = {}
for line in open("states_local.jsonl", encoding="utf-8"):
    row = json.loads(line)
    states[(row.get("dataset", "default"), str(row["item_id"]))] = row["state"]

rows = {}
for item in manifest["items"]:
    key = (item["dataset"], item["item_id"])
    # dataset and item_id let `s1 trainset record` check this file against the manifest
    row = rows.setdefault(key, {"dataset": key[0], "item_id": key[1], "state": states[key], "questions": {}})
    row["questions"][item["question_id"]] = {**wording[item["question_id"]], "label": item["label"]}

with open("train.jsonl", "w", encoding="utf-8") as f:
    for row in rows.values():
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

**The training record** (`s1-trainrun/1`, from `s1 trainset record`) ties a checkpoint's name to what produced it: the manifest's SHA-256, the data file's SHA-256 with its row and item counts, the SHA-256 of each `--weights` file by name, and, as you pass them, flat trainer settings (`--trainer`), summary metrics (`--summary`) and per-step metrics (`--steps`, JSONL with an increasing integer `step`). `--parent-run` links the record of the run a checkpoint started from. `trainrun_id` hashes everything but `created_at`, so the same inputs give the same id, and a record whose contents no longer match its id is refused as a parent. Before writing, it checks that the data file's `(dataset, item_id)` set is the manifest's: an item the manifest does not list exits 2, since the refusal below would not know the checkpoint trained on it; a listed item with no row exits 1; an inherited item may appear (a replay). Other fields of a data row, `state` and labels among them, are neither read nor kept, and a trainer setting or summary metric named `state` is refused (exit 1), as the ledger refuses that key: `--training-run` copies these values into reports and the Langfuse export. s1ops parses no trainer's log: convert it to these files first.

**Showing the run.** `--training-run` is repeatable on `s1 compare`, `s1 policy tune`, `s1 policy curve` and `s1 export langfuse`, one record per checkpoint. A model is matched to a record by name: `model_served` equal to the record's `checkpoint`. `s1 compare` adds `primary_trainrun_id` and `shadow_trainrun_id` to each question's report; `s1 policy curve` adds `trainrun_id` to each group, and `s1 policy tune` to each policy of its printed summary. The value is `null` for a model no record names. `s1 policy tune` also writes `trainrun_id` into `policy.json`, on each policy whose model a record names and on no other; it says where the policy came from, like `source_sha256`, is no part of `policy_sha256`, and nothing routes on it. s1ops 0.11.1 and earlier refuse a policy file that has the field, so tune without the option for a reader that old; a policy file tuned without it is what it always was. `s1 export langfuse` writes each record, as it is on disk, into a `training_runs` list ordered by checkpoint, counts them in `summary.n_training_runs`, and puts `metadata.trainrun_id` on every observation that checkpoint answered; a record's settings and metrics are copied as `s1 trainset record` was given them: none can be named `state`, and keeping other text out of `--trainer` and `--summary` is yours to do. Without the option, every one of these outputs is what it was. A record whose checkpoint answered no record in the input exits 1, as a manifest does, and so do two different records for one checkpoint, a record whose contents no longer match its `trainrun_id`, and, in `s1 export langfuse --policy`, a record of another run than the one that model's policy names (two sets of weights under one name). The option shows a link and checks nothing about leakage: that is `--training-manifest`, which it neither replaces nor is compared with. `s1 export phoenix` does not take it. In the SDK: `compare_records(..., training_runs=[path])` and `build_langfuse_events(..., training_runs=[load_trainrun(path)])`.

**The refusal.** `--training-manifest` is repeatable on `s1 fit`, `s1 audit`, `s1 policy tune`, `s1 policy curve` and `s1 compare`. It exits 2 when a group those filters select is served under the manifest's checkpoint and holds an item it trained on; `s1 compare` checks each pair against the model of its side, and a candidate's training item the other file lacks is not compared. An item is its `(dataset, item_id)` across questions, because the model saw the state and rewording a question changes its `question_id`. Another model's answers on those items pass. A manifest whose checkpoint answered no record in the input file exits 1, which catches a misspelled name; it does not promise that the rows selected include that model, so `--model current-1` on a mixed file checks no candidate row.

**What it does not cover:**

- The check runs only in a command given the flag. `s1 apply`, `s1 drift`, `s1 slice`, `s1 labelaudit` and the workbench do not take it, and `s1 policy tune --use-calib` does not look behind a stored `calib`: an `s1 fit` on trained items without the flag, followed by `s1 apply` on clean items, passes a checked `tune`. `fits.json` does not record that a check ran.
- `s1 trainset record` checks the data file you name, not what the trainer read: pass the file the trainer was given.
- `--parent` covers ancestors exported through s1ops. A released base checkpoint's training set is unknown here.
- Identity is exact. A near-duplicate or another item from a trained source is not refused, though it can inflate a score the same way.
- `--model` is your claim about what answers. A server such as `kev.serve` echoes the requested name without checking it, so name every checkpoint differently: fits, policies and drift's model comparison key on that name and see two checkpoints under one name as one model. A name is 1-128 characters, an ASCII letter or digit and then ASCII letters, digits or `._:/@+-`, the form a live backend accepts as `model_served`. `s1 trainset export` reports `answered_as_checkpoint` when the source file already holds records under the name.

---

## CLI Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Usage error, missing or malformed input, no matching groups, `lint check --strict` findings, policy fingerprint mismatch, ledger error, `s1 demo` target not new or empty |
| `2` | In-sample refusal: `s1 apply`, or `s1 drift` or `s1 export` with `--fit`, on an item the fit used (override with `--allow-in-sample`), or `--training-manifest` on an item the checkpoint trained on, or `s1 trainset record` on a data item its manifest does not list |
| `3` | `model_served` mismatch against the fit or policy (`s1 apply` without `--pass-through-unfitted`, `s1 gate`) |
| `4` | `s1 drift --fail-on-critical` found a `CRITICAL` question |

`s1 promote` exits with the code of the step that refused (`2` for a trained item or for calibrating on the evaluation fold, `3` for a `model_served` mismatch), and names the step. Errors print a single `s1: <message>` line on stderr; non-printable characters in it (a newline or an escape sequence inside an id or a path) are written escaped, as `\n` or `\x1b`, while printable text, non-ASCII included, is kept. When an input that another command produces is missing, the message names that command (for example `run \`s1 fit\` first`). Set `S1_DEBUG=1` to re-raise errors raised by the library, with their traceback; checks the CLI makes itself (a missing input file, an empty grades file) still print one line. A missing input path is an error everywhere, including `s1 ledger status`, except `s1 lint check`, which reports an unreadable questions file as a `JL000` finding in its normal (`--format json`) report.

---

## Development

From a source checkout (the sdist includes the test suite), all tests run without API keys or calls to external services (a few backend tests talk to an HTTP server on `127.0.0.1`):

```bash
uv sync --all-extras
uv run pytest -v
```
