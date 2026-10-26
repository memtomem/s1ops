# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`s1ops` is an offline-first, vendor-neutral LLMOps SDK + CLI for **System 1 decision models**: backends that return a calibrated probability distribution rather than text, in one of three primitives — `noul` (binary `p`), `choice` (K options), `score` (ordinal 0..L-1). It covers question linting, post-hoc calibration, threshold/escalation-band compilation, slice and blind label audits, shadow/judge comparison, drift monitoring, a runtime `DecisionGate`, `safe-v1` exporters (Langfuse, Phoenix/OpenInference), and a local FastAPI web workbench.

Design docs (in `docs/`, written in Korean; README is English):
- `docs/ARCHITECTURE_AND_PLAN.md` — data contracts, module map, Phase 1–3 (all done).
- `docs/NEXT_STEPS_PLAN.md` — Phase 4 open work: CI, live HTTP backends, `DecisionGate` choice/score, TS SDK.
- `docs/REVIEW_AND_COMPARISON.md` — why existing System 2 tools don't fit, failure modes the guardrails encode, naming decision.

## Commands

```bash
uv sync --all-extras                          # install (extras: langfuse, typesafe)
uv run pytest                                 # full suite, fully offline, no API keys
uv run pytest tests/test_drift_runner_and_gate.py::test_runtime_decision_gate_and_cli_web
uv run s1 demo examples/data                  # seeded demo workspace (new or empty dir only)
uv run s1 web --workspace examples/data --host 127.0.0.1 --port 8765
```

The full end-to-end CLI pipeline over the demo workspace (lint → fit → apply → audit → policy tune → slice → labelaudit → compare/judge → drift → probe/gate → export) is spelled out in README's Quickstart §3; later steps consume `fits.json`/`policy.json` produced by earlier ones.

No linter or type checker is configured in `pyproject.toml`. CI (`.github/workflows/ci.yml`) runs the locked suite on Python 3.11–3.13 and `scripts/release_check.py` on the built wheel and sdist; `release.yml` publishes by manual dispatch on a `vX.Y.Z` tag (procedure in `docs/NEXT_STEPS_PLAN.md` Track 1). The repository is private while the package is public, so package metadata must not name it.

## Architecture

**Package layout.** `src/s1ops/` is the real package; `src/s1/__init__.py` is only a star-re-export alias so `import s1` works. Both `s1` and `s1ops` console scripts point at `s1ops.cli:main`, a single argparse file with every subcommand. The demo generator lives in the package (`s1ops/demo.py`, behind `s1 demo`) so it works from `pip install`; tests build their fixture workspace into `tmp_path` with `from s1ops.demo import generate_workspace` rather than using checked-in data. Its `fitted_at`/`created_at` fields are wall-clock, so compare workspaces with those dropped. `pytest` runs with `pythonpath = ["src", "."]` so `tests/test_release_check.py` can import `scripts.release_check`. `examples/data/` is gitignored output.

**Data flow.** `contracts/` defines everything else's inputs: `questions.py` (Noul/Choice/Score models, `question_id`, `/v1/decisions` conversion), `records.py` (strict `jev-record/1` / `s1-record/1` parser and grouping), `schemas.py` (`PolicyEntry`, `s1-policy/1`). `improve/calibrate.py` writes `fits.json`; `improve/policy.py` writes `policy.json`; `runtime/gate.py` (`DecisionGate`) loads both at runtime. `monitor/runner.py` defines the `Backend` protocol and `DeterministicMockBackend`, and runs probes under `ledger/journal.py`; `monitor/backends.py` has `HttpSystemOneBackend` (wraps `typesafe-sdk`, the `typesafe` extra) and `OpenAIDecisionsBackend` (wraps `openai`, the `openai` extra), both imported lazily; their SDK and transport logs (`QUIET_LOGGERS`) are dropped during `infer`, and a new logger that speaks during a request fails `tests/test_openai_backend.py`. `web/app.py` serves `web/static/index.html` and computes tab data from a workspace directory.

**Invariants that cut across modules** — preserve them when changing anything:
- **Group key is `(question_id, model_served, primitive)`.** Metrics, fits, policies, drift, and exports are never pooled across groups. Changing a question's wording or option order changes its `question_id`.
- **`question_id`** = `q-` + first 16 hex of SHA-256 over the normalized `/v1/systemone` wire JSON (option order preserved).
- **`safe-v1` zero-state:** raw item `state` text is never written to records, ledgers, probe output, or exports — only `(dataset, item_id)` and `state_sha256`. A local `states_local.jsonl` (`--states`) is read by two commands only: `s1 probe` sends each state to the backend, and `s1 labelaudit sheet` writes the state text of every reviewed item (top flags plus random controls) into the review sheet, which is meant for a human reviewer.
- **Record validation** rejects NaN/Inf, JSON bools posing as numbers, and probability sums outside `1 ± 0.02`, raising `RecordError` with a 1-based line number.
- **Out-of-sample enforcement:** `fits.json` records every `[dataset, item_id]` used in fitting plus `source_sha256`. `s1 apply` exits **2** on an in-sample item (unless `--allow-in-sample`) and **3** on a `model_served` mismatch; `DecisionGate` raises `ModelServedMismatchError`.
- **`policy_sha256`** fingerprints each `PolicyEntry`'s routing fields. Checking it is opt-in: `evaluate_*(expected_policy_sha256=...)` / `s1 gate --expected-sha256` raise `PolicyIntegrityError` on mismatch. `load_policies` recomputes the fingerprint and does not compare it with the stored one.
- **`policy.json` / `fits.json` loading** validates through strict pydantic row models (`_PolicyRow` in `contracts/schemas.py`, `_FitRow` in `improve/calibrate.py`) and then builds the frozen dataclasses. A malformed file raises `ValueError` / `FitError` naming the file and the field path (`policies.0.threshold`); never let a `TypeError`/`AttributeError` escape a loader.
- **Thresholds are tie-safe:** System 1 APIs return rounded probabilities, so `select_threshold` accepts/rejects whole equal-probability groups atomically.
- **Ledger is reserve-then-settle** with exclusive `flock` on `<journal>.lock`, rejects hard-linked ledgers, and halts with `LedgerUnusable` if a measured cost exceeds its reservation (`settle` only records the overrun; `run_batch` raises right after it). `run_batch` reserves per attempt and settles `measured` when a cost is reported, `unsent` only when the request provably never left, `unknown` otherwise — never `unsent` for a request that may have been billed. `BackendError` messages carry an error code only, never server text (a server may echo `state`).
- **Lint rules** `JL000`–`JL060` (a range with gaps, not 61 rules) can be suppressed via `x-s1-lint` / `x-jev-lint` keys, except `UNSUPPRESSIBLE` (`JL000`, `JL001`, `JL003`, `JL020`).

**CLI error contract** (`cli.py`, tested in `tests/test_cli_errors.py`). Exit codes: 0 ok, 1 any user-facing failure (including argparse usage errors, via `_Parser`), 2 `apply` in-sample refusal, 3 `model_served` mismatch (`apply`, `gate`), 4 `drift --fail-on-critical`. Commands raise `CliError(message, code)` and `main()` prints one `s1: <message>` line; it also maps `OSError`, `ValueError` (covers `RecordError`/`FitError`/`LedgerCorrupt`/`JSONDecodeError`), `PolicyIntegrityError`, `BudgetExceeded`, and `ModelServedMismatchError`. `KeyError` is deliberately not mapped — convert it at the call site. `S1_DEBUG=1` re-raises library exceptions for a traceback; `CliError` always prints one line. Input paths are pre-checked by dest name via `_INPUTS`, which also names the producing command (`run \`s1 fit\` first`); a new input argument needs an `_INPUTS` entry, and an output path must not reuse an input dest name. `lint check` is exempt: it reports an unreadable file as a `JL000` finding inside its own report. Every argument needs `help=` (a test enforces it), and each `add_parser` must pass `**_FMT`, since argparse does not inherit `formatter_class`.

**Web workbench errors** (`tests/test_web_errors.py`). Each file read goes through `_load_or_400` (catches `ValueError` only), so a malformed workspace file is HTTP 400 naming that file. Never wrap a computation in it: computations get pre-loaded records and run unwrapped so a server bug stays a 500; an input condition only detectable inside a computation gets its own exception type (`CompareInputError` in `eval/compare.py`) caught explicitly. Undefined judge kappa is `None` in `Agreement.as_dict()` (NaN is not valid JSON) and renders as "N/A". A missing optional file is still a 200 with empty data, and `/api/overview` lists it in `missing_files`. In `index.html`, every server-derived string passes through `esc()` before `innerHTML`; a test scans the template literals for unescaped identifier fields.

**Routing.** `runtime/routing.py::route` is the only decision rule for a `PolicyEntry`: `DecisionGate` (`evaluate_noul` / `evaluate_choice` / `evaluate_score`), drift's escalation rate and both exporters' `policy_action` call it, and callers only choose which distribution (raw or calibrated) to route. It dispatches on `policy.primitive`, never on `classes` (`(0, 1) == (False, True)`). Choice/score: `error_budget` only, a tie at the top or `threshold: null` escalates, and the gate refuses a calibrated policy without its fit. Distributions are renormalized with `records.probability_total` (`math.fsum`) in the parser and the gate, so a tuned threshold compares bit-identically at runtime on every Python (plain `sum` is order-dependent on 3.11). Exports of policy-routed records carry `routing_rule` (`ROUTING_RULE`); bump it when the rule's outputs change.

## Related repos

This SDK consolidates prototypes from `jev-utils` (`/Users/pdstudio/work/jev-utils`) and keeps `jev-record/1` / `jev-calib-fit/2` compatibility. Downstream consumers: `decision-plane` (loads `fits.json` + `policy.json`) and `tracegraph` (ingests `s1 export phoenix` spans), both under `/Users/pdstudio/Work/agent-harness/`.
