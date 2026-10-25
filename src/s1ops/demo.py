"""Deterministic synthetic demo workspace generator, behind `s1 demo`.

Generates a complete, 100% offline System 1 evaluation workspace:
- `questions.json`: System 1 questions (including intentional JL010/JL011/JL014/JL023 lint cases)
- `records_train.jsonl`: 120 calibration fold records (`jev-1.13.0`)
- `records.jsonl`: 160 held-out evaluation records (`jev-1.13.0`, calibrated via `fits.json`)
- `records_kev.jsonl`: 160 paired shadow records (`kev-1.0`) for McNemar-Holm comparison
- `records_canary.jsonl`: 120 canary records with distribution drift and a silent `model_served` upgrade
- `probe_records.jsonl`: Noul/Choice twin & inverse coherence probes (`probe_role`: `noul`, `twin`, `inverse`)
- `features.jsonl`: per-item slice & feature tags (answerability + BH-FDR error slice mining)
- `states_local.jsonl`: local-only synthetic item states for blind review sheet generation
- `judge_grades.jsonl`: rubric judge vs human grades (0..4) across 3 criteria
- `fits.json` + `policy.json`: compiled calibration and routing policy artifacts
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from s1ops.contracts.questions import ChoiceQuestion, NoulQuestion, question_id
from s1ops.contracts.records import parse_record, read_records, select
from s1ops.contracts.schemas import PolicyEntry, save_policies
from s1ops.eval.metrics import top_label
from s1ops.improve.calibrate import apply_to_record, fit_group, save_fits
from s1ops.improve.policy import select_threshold


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_workspace(out_dir: str | Path, *, seed: int = 42) -> dict[str, str]:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Canonical questions + lint demonstration questions
    q_noul = NoulQuestion(instructions="Does the user message request a financial wire transfer?")
    q_choice = ChoiceQuestion(
        instructions="What is the primary support routing intent of the message?",
        criteria={
            "billing": "Billing or invoice payment questions",
            "technical": "Technical troubleshooting or bug reports",
            "account": "Account login or profile settings",
        },
    )
    q_twin_choice = ChoiceQuestion(
        instructions="Does the user message request a financial wire transfer?",
        criteria={"yes": "Requests a wire transfer", "no": "Does not request a wire transfer"},
    )
    q_inv_noul = NoulQuestion(
        instructions="Is the user message unrelated to a financial wire transfer?"
    )

    qid_noul = question_id(q_noul)
    qid_twin = question_id(q_twin_choice)
    qid_inv = question_id(q_inv_noul)

    questions_doc = {
        "wire_transfer": q_noul.wire_dict(),
        "support_intent": q_choice.wire_dict(),
        # Intentional lint examples for JL010, JL020, JL030, JL040
        "bad_negation_compound": {
            "type": "noul",
            "instructions": "Is the response not unsafe and does it also avoid policy violations or bad tone?",
        },
        "bad_hedge": {
            "type": "noul",
            "instructions": "Is the user message usually appropriate for general audiences without major issues?",
        },
        "bad_choice_yes_no": {
            "type": "choice",
            "instructions": "Is this document relevant?",
            "criteria": {"yes": "Relevant", "no": "Not relevant"},
        },
    }
    q_path = out / "questions.json"
    q_path.write_text(json.dumps(questions_doc, indent=2) + "\n", encoding="utf-8")

    # 2. Calibration fold (records_train.jsonl) for jev-1.13.0 on qid_noul
    train_objs: list[dict[str, Any]] = []
    for i in range(120):
        item_id = f"cal-{i:04d}"
        state_text = f"[CAL {i}] Synthetic wire transfer inquiry sample #{i}"
        z_true = rng.gauss(0.0, 1.35)
        p_true = _sigmoid(z_true)
        y = rng.random() < p_true
        # Overconfident model: multiplies logit by 1.9
        p_raw = min(max(_sigmoid(z_true * 1.9 + rng.gauss(0.0, 0.15)), 0.001), 0.999)
        train_objs.append(
            {
                "schema": "s1-record/1",
                "dataset": "fin-guard-cal",
                "split": "cal",
                "item_id": item_id,
                "state_sha256": _sha256(state_text),
                "question_id": qid_noul,
                "primitive": "noul",
                "model_requested": "jev-1.13.0",
                "model_served": "jev-1.13.0",
                "p": round(p_raw, 6),
                "label": bool(y),
                "latency_ms": round(rng.uniform(18.0, 38.0), 2),
                "timestamp": "2026-10-25T00:00:00Z",
            }
        )

    train_path = out / "records_train.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for obj in train_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    # Fit temperature scaling on train fold
    train_recs = read_records(train_path)
    sel_train = select(train_recs)
    grp_noul = sel_train.groups[0]
    fit_noul = fit_group(grp_noul, method="temperature", split="cal", min_n=30)
    fits_path = out / "fits.json"
    save_fits([fit_noul], fits_path)

    # 3. Held-out evaluation fold (records.jsonl for jev-1.13.0 + records_kev.jsonl for kev-1.0)
    eval_objs: list[dict[str, Any]] = []
    kev_objs: list[dict[str, Any]] = []
    probe_objs: list[dict[str, Any]] = []
    features_objs: list[dict[str, Any]] = []
    states_objs: list[dict[str, Any]] = []

    for i in range(160):
        item_id = f"eval-{i:04d}"
        if i < 110:
            slice_tag = "answerable"
        elif i < 140:
            slice_tag = "unanswerable"
        else:
            slice_tag = "adversarial_trap"

        lang = "ko_slang" if (i % 6 == 0) else ("ko_formal" if i % 2 == 0 else "en")
        has_code = (i % 9 == 0)
        state_text = (
            f"[EVAL {i} | slice={slice_tag} | lang={lang}] "
            f"Customer message regarding wire transfer or account routing #{i}."
        )
        s_hash = _sha256(state_text)
        states_objs.append(
            {"dataset": "fin-guard-eval", "item_id": item_id, "state": state_text}
        )
        features_objs.append(
            {
                "dataset": "fin-guard-eval",
                "item_id": item_id,
                "slice": slice_tag,
                "features": {
                    "slice": slice_tag,
                    "lang": lang,
                    "has_code": has_code,
                },
            }
        )

        z_true = rng.gauss(0.1, 1.4)
        p_true = _sigmoid(z_true)
        y_gold = rng.random() < p_true

        # Inject 6 intentional gold label flips on high-confidence items for labelaudit
        is_flipped_label = i in (3, 19, 41, 73, 105, 131)
        if is_flipped_label:
            z_true = 2.6 if i % 2 == 0 else -2.6
            y_true_latent = z_true > 0
            y_gold = not y_true_latent
        else:
            y_true_latent = y_gold

        # Inject systematic error slice on lang=ko_slang for jev-1.13.0
        if lang == "ko_slang" and not is_flipped_label:
            z_jev = (-1.6 if y_true_latent else 1.6) + rng.gauss(0.0, 0.25)
        elif slice_tag == "unanswerable":
            z_jev = rng.gauss(0.0, 0.25)
        else:
            z_jev = (1.8 if y_true_latent else -1.8) + rng.gauss(0.0, 0.65)

        p_raw_jev = min(max(_sigmoid(z_jev * 1.85), 0.001), 0.999)
        base_rec_dict = {
            "schema": "s1-record/1",
            "dataset": "fin-guard-eval",
            "split": "eval",
            "item_id": item_id,
            "state_sha256": s_hash,
            "question_id": qid_noul,
            "primitive": "noul",
            "model_requested": "jev-1.13.0",
            "model_served": "jev-1.13.0",
            "p": round(p_raw_jev, 6),
            "label": bool(y_gold),
            "latency_ms": round(rng.uniform(19.0, 41.0), 2),
            "timestamp": "2026-10-25T01:00:00Z",
        }
        parsed_base = parse_record(base_rec_dict)
        cal_rec = apply_to_record(fit_noul, parsed_base)
        eval_objs.append(cal_rec)

        # Paired kev-1.0 record
        z_kev = (1.6 if y_true_latent else -1.6) + rng.gauss(0.0, 0.85)
        p_raw_kev = min(max(_sigmoid(z_kev), 0.001), 0.999)
        kev_objs.append(
            {
                "schema": "s1-record/1",
                "dataset": "fin-guard-eval",
                "split": "eval",
                "item_id": item_id,
                "state_sha256": s_hash,
                "question_id": qid_noul,
                "primitive": "noul",
                "model_requested": "kev-1.0",
                "model_served": "kev-1.0",
                "p": round(p_raw_kev, 6),
                "label": bool(y_gold),
                "latency_ms": round(rng.uniform(32.0, 64.0), 2),
                "timestamp": "2026-10-25T01:00:00Z",
            }
        )

        # Probe coherence records for first 50 items
        if i < 50:
            probe_objs.append(
                {
                    **base_rec_dict,
                    "probe_role": "noul",
                    "probe_question": "wire_transfer",
                }
            )
            p_twin = min(max(p_raw_jev + rng.gauss(0.0, 0.035), 0.001), 0.999)
            probe_objs.append(
                {
                    "schema": "s1-record/1",
                    "dataset": "fin-guard-eval",
                    "split": "eval",
                    "item_id": item_id,
                    "state_sha256": s_hash,
                    "question_id": qid_twin,
                    "primitive": "choice",
                    "model_requested": "jev-1.13.0",
                    "model_served": "jev-1.13.0",
                    "choice": "yes" if p_twin >= 0.5 else "no",
                    "probabilities": {"yes": round(p_twin, 6), "no": round(1.0 - p_twin, 6)},
                    "confidence": round(max(p_twin, 1.0 - p_twin), 6),
                    "label": "yes" if y_gold else "no",
                    "probe_role": "twin",
                    "probe_question": "wire_transfer",
                }
            )
            p_inv = min(max((1.0 - p_raw_jev) + rng.gauss(0.02, 0.04), 0.001), 0.999)
            probe_objs.append(
                {
                    **base_rec_dict,
                    "question_id": qid_inv,
                    "primitive": "noul",
                    "p": round(p_inv, 6),
                    "label": not bool(y_gold),
                    "probe_role": "inverse",
                    "probe_question": "wire_transfer",
                }
            )

    rec_path = out / "records.jsonl"
    with rec_path.open("w", encoding="utf-8") as fh:
        for obj in eval_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    kev_path = out / "records_kev.jsonl"
    with kev_path.open("w", encoding="utf-8") as fh:
        for obj in kev_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    probe_path = out / "probe_records.jsonl"
    with probe_path.open("w", encoding="utf-8") as fh:
        for obj in probe_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    feat_path = out / "features.jsonl"
    with feat_path.open("w", encoding="utf-8") as fh:
        for obj in features_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    states_path = out / "states_local.jsonl"
    with states_path.open("w", encoding="utf-8") as fh:
        for obj in states_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    # 4. Compile policy.json from calibrated evaluation fold
    eval_recs = read_records(rec_path, use_calib=True)
    sel_eval = select(eval_recs)
    grp_eval = sel_eval.groups[0]
    conf, correct = top_label(grp_eval.P, grp_eval.y)
    tau = select_threshold(conf, correct, budget=0.10)
    policy_entry = PolicyEntry(
        question_id=qid_noul,
        model_served="jev-1.13.0",
        primitive="noul",
        mode="escalation_band",
        split="eval",
        n=grp_eval.n,
        threshold=tau,
        error_budget=0.10,
        band_lo=0.25,
        band_hi=0.75,
        calibrated=True,
    )
    policy_path = out / "policy.json"
    save_policies([policy_entry], policy_path)

    # 5. Judge grades (judge_grades.jsonl, 0..4 scale)
    judge_objs: list[dict[str, Any]] = []
    criteria = ["faithfulness", "policy_compliance", "tone"]
    for i in range(90):
        crit = criteria[i % len(criteria)]
        human = rng.randint(1, 4)
        delta = rng.choices([-1, 0, 0, 0, 1], weights=[10, 25, 35, 15, 15])[0]
        judge = min(max(human + delta, 0), 4)
        judge_objs.append(
            {
                "item_id": f"j-{i:04d}",
                "criterion": crit,
                "human": human,
                "judge": judge,
            }
        )
    judge_path = out / "judge_grades.jsonl"
    with judge_path.open("w", encoding="utf-8") as fh:
        for obj in judge_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    # 6. Canary production window (records_canary.jsonl: silent model_served upgrade + PSI shift)
    canary_objs: list[dict[str, Any]] = []
    for i in range(120):
        item_id = f"can-{i:04d}"
        s_hash = _sha256(f"[CANARY {i}] Wire transfer canary traffic #{i}")
        # Silent provider upgrade on 30% of traffic + upward probability shift
        served = "jev-1.14.0-rc1" if (i % 3 == 0) else "jev-1.13.0"
        p_shifted = min(max(_sigmoid(rng.gauss(1.15, 0.9)), 0.001), 0.999)
        canary_objs.append(
            {
                "schema": "s1-record/1",
                "dataset": "fin-guard-prod-canary",
                "split": "prod",
                "item_id": item_id,
                "state_sha256": s_hash,
                "question_id": qid_noul,
                "primitive": "noul",
                "model_requested": "jev-1.13.0",
                "model_served": served,
                "p": round(p_shifted, 6),
                "latency_ms": round(rng.uniform(28.0, 72.0), 2),
                "timestamp": "2026-10-25T03:00:00Z",
            }
        )
    canary_path = out / "records_canary.jsonl"
    with canary_path.open("w", encoding="utf-8") as fh:
        for obj in canary_objs:
            fh.write(json.dumps(obj, sort_keys=True) + "\n")

    return {
        "questions": str(q_path),
        "records_train": str(train_path),
        "records": str(rec_path),
        "records_kev": str(kev_path),
        "records_canary": str(canary_path),
        "probe_records": str(probe_path),
        "features": str(feat_path),
        "states_local": str(states_path),
        "fits": str(fits_path),
        "policy": str(policy_path),
        "judge_grades": str(judge_path),
    }
