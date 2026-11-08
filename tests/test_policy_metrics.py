"""`metrics_on_split` says what a policy does on the split it was tuned on, by the one routing rule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import Group
from s1ops.contracts.schemas import PolicyEntry, load_policies
from s1ops.demo import generate_workspace
from s1ops.runtime.routing import routed_metrics


def _group(classes: tuple, rows: list[list[float]], labels: list[int], primitive: str) -> Group:
    return Group(
        key=("q-0000000000000000", "m", primitive),
        classes=classes,
        P=np.array(rows, dtype=float),
        y=np.array(labels),
        items=tuple((None, f"i{n}") for n in range(len(rows))),
        splits=tuple(None for _ in rows),
    )


def _policy(primitive: str, **kw) -> PolicyEntry:
    base = dict(question_id="q-0000000000000000", model_served="m", primitive=primitive, mode="error_budget", split="eval", n=1, threshold=0.8, error_budget=0.1)
    return PolicyEntry(**{**base, **kw})


def _counts(metrics: dict) -> tuple:
    return metrics["n"], metrics["accepted"], metrics["errors"], metrics["coverage"], metrics["accepted_error"]


def test_noul_error_budget_counts_allow_and_block() -> None:
    # 0.9 ALLOW right, 0.95 ALLOW wrong, 0.1 BLOCK right, 0.05 BLOCK wrong, 0.6 escalates.
    group = _group((False, True), [[0.1, 0.9], [0.05, 0.95], [0.9, 0.1], [0.95, 0.05], [0.4, 0.6]], [1, 0, 0, 1, 1], "noul")
    assert _counts(routed_metrics(_policy("noul"), group)) == (5, 4, 2, 0.8, 0.5)


def test_noul_band_routes_by_the_band_not_the_top_label() -> None:
    # In band [0.2, 0.4): 0.45 is ALLOW and its label is false, an error though the top label is false too.
    group = _group((False, True), [[0.55, 0.45], [0.9, 0.1], [0.7, 0.3], [0.1, 0.9]], [0, 1, 0, 1], "noul")
    policy = _policy("noul", mode="escalation_band", band_lo=0.2, band_hi=0.4)
    metrics = routed_metrics(policy, group)
    assert _counts(metrics) == (4, 3, 2, 0.75, 2 / 3)
    assert metrics["selection"] == "escalation_band"


def test_choice_counts_a_wrong_top_answer() -> None:
    group = _group(("a", "b"), [[0.9, 0.1], [0.85, 0.15], [0.3, 0.7]], [0, 1, 1], "choice")
    assert _counts(routed_metrics(_policy("choice"), group)) == (3, 2, 1, 2 / 3, 0.5)


def test_a_tie_at_or_above_the_threshold_escalates() -> None:
    group = _group(("a", "b"), [[0.5, 0.5], [0.6, 0.4]], [0, 0], "choice")
    assert _counts(routed_metrics(_policy("choice", threshold=0.5), group)) == (2, 1, 0, 0.5, 0.0)


def test_score_compares_levels() -> None:
    group = _group((0, 1, 2), [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.4, 0.3, 0.3]], [0, 1, 2], "score")
    assert _counts(routed_metrics(_policy("score"), group)) == (3, 2, 1, 2 / 3, 0.5)


def test_no_threshold_accepts_nothing_and_has_no_error_rate() -> None:
    group = _group(("a", "b"), [[0.9, 0.1], [0.1, 0.9]], [0, 1], "choice")
    assert _counts(routed_metrics(_policy("choice", threshold=None), group)) == (2, 0, 0, 0.0, None)


def test_cp_alpha_is_recorded_only_when_it_was_used() -> None:
    group = _group(("a", "b"), [[0.9, 0.1]], [0], "choice")
    policy = _policy("choice")
    assert routed_metrics(policy, group, use_cp_upper=False, cp_alpha=float("nan"))["cp_alpha"] is None
    used = routed_metrics(policy, group, use_cp_upper=True, cp_alpha=0.05)
    assert (used["use_cp_upper"], used["cp_alpha"]) == (True, 0.05)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def test_tune_writes_metrics_a_strict_reader_loads_back(ws, tmp_path, capsys) -> None:
    out = tmp_path / "policy.json"
    # An unused --cp-alpha nan is accepted without --use-cp-upper, and must not reach the file.
    assert cli_main(["policy", "tune", ws["records"], "--error-budget", "0.1", "--cp-alpha", "nan", "-o", str(out)]) == 0
    json.loads(out.read_text(encoding="utf-8"), parse_constant=lambda name: pytest.fail(f"{name} is not JSON"))
    (policy,) = load_policies(out)
    metrics = policy.metrics_on_split
    assert metrics["selection"] == "escalation_band" and metrics["cp_alpha"] is None
    assert metrics["n"] == policy.n == 160
    assert metrics["accepted"] == round(metrics["coverage"] * metrics["n"])


def _counted_from_rows(policy: PolicyEntry, path: str) -> tuple[int, int]:
    """Accepted and errors read straight from the file, sharing no code with the routing module."""
    accepted = errors = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["question_id"] != policy.question_id or row["model_served"] != policy.model_served or row.get("label") is None:
            continue
        source = row["calib"] if policy.calibrated else row
        if policy.primitive == "noul":
            p = source["p"]
            if policy.mode == "escalation_band":
                if policy.band_lo <= p < policy.band_hi:
                    continue
                answer = p >= policy.band_hi
            else:
                if policy.threshold is None or max(p, 1 - p) < policy.threshold:
                    continue
                answer = p >= 0.5
            label = row["label"]
        else:
            total = math.fsum(source["probabilities"].values())
            probabilities = {k: v / total for k, v in source["probabilities"].items()}
            top = max(probabilities.values())
            leaders = [k for k, v in probabilities.items() if v == top]
            if len(leaders) > 1 or policy.threshold is None or top < policy.threshold:
                continue
            kind = int if policy.primitive == "score" else str  # a score key may be "01"
            answer, label = kind(leaders[0]), kind(row["label"])
        accepted += 1
        errors += answer != label
    return accepted, errors


@pytest.mark.parametrize(
    "records, args",
    [
        pytest.param("records", ("--error-budget", "0.1"), id="noul-band-raw"),
        pytest.param("records", ("--error-budget", "0.1", "--use-calib"), id="noul-band-calibrated"),
        pytest.param("records", ("--error-budget", "0.1", "--max-escalation", "0"), id="noul-empty-band-raw"),
        pytest.param("records_categorical", ("--error-budget", "0.1"), id="choice-and-score-raw"),
        pytest.param("records_categorical", ("--error-budget", "0.1", "--use-calib"), id="choice-and-score-calibrated"),
    ],
)
def test_tune_metrics_match_a_count_from_the_file(ws, tmp_path, capsys, records, args) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws[records], *args, "-o", str(out)]) == 0
    policies = load_policies(out)
    assert policies
    for policy in policies:
        accepted, errors = _counted_from_rows(policy, ws[records])
        metrics = policy.metrics_on_split
        assert (metrics["accepted"], metrics["errors"]) == (accepted, errors)
        assert metrics["coverage"] == accepted / metrics["n"]
        assert metrics["accepted_error"] == (errors / accepted if accepted else None)
        assert accepted > 0


@pytest.mark.parametrize(
    "primitive, probabilities, label",
    [
        pytest.param("choice", {"a": 0.60, "b": 0.39}, "a", id="sums-to-0.99"),
        pytest.param("score", {"00": 0.9, "01": 0.1}, 0, id="score-keys-with-leading-zeros"),
    ],
)
def test_the_file_count_reads_a_row_as_the_parser_does(tmp_path, capsys, primitive, probabilities, label) -> None:
    records = tmp_path / "r.jsonl"
    row = {"schema": "s1-record/1", "dataset": "d", "item_id": "i", "question_id": "q-0000000000000000", "primitive": primitive,
           "model_requested": "m", "model_served": "m", "probabilities": probabilities, "label": label,
           "confidence": 0.9, **({"choice": "a"} if primitive == "choice" else {"score": 0, "expected_score": 0.1})}
    records.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", str(records), "--error-budget", "0.1", "-o", str(out)]) == 0
    (policy,) = load_policies(out)
    assert (policy.metrics_on_split["accepted"], policy.metrics_on_split["errors"]) == _counted_from_rows(policy, str(records)) == (1, 0)


@pytest.mark.parametrize("policy_key, records", [("policy", "records"), ("policy_categorical", "records_categorical")])
def test_demo_metrics_match_a_count_from_the_file(ws, policy_key, records) -> None:
    for policy in load_policies(ws[policy_key]):
        accepted, errors = _counted_from_rows(policy, ws[records])
        assert (policy.metrics_on_split["accepted"], policy.metrics_on_split["errors"]) == (accepted, errors)
        assert accepted > 0


def test_a_band_can_exceed_the_budget_and_the_file_shows_it(ws, tmp_path, capsys) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws["records"], "--error-budget", "0.1", "-o", str(out)]) == 0
    (policy,) = load_policies(out)
    assert policy.mode == "escalation_band" and policy.metrics_on_split["selection"] == "escalation_band"
    assert policy.metrics_on_split["accepted_error"] > policy.error_budget


def test_an_empty_band_accepts_every_item_as_the_readme_says(ws, tmp_path, capsys) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws["records"], "--error-budget", "0.1", "--max-escalation", "0", "-o", str(out)]) == 0
    (policy,) = load_policies(out)
    assert policy.mode == "escalation_band"
    assert policy.metrics_on_split["accepted"] == policy.n and policy.metrics_on_split["coverage"] == 1.0


def test_tune_records_the_cp_settings_it_used(ws, tmp_path, capsys) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws["records_categorical"], "--error-budget", "0.1", "--use-cp-upper", "--cp-alpha", "0.1", "-o", str(out)]) == 0
    for policy in load_policies(out):
        assert (policy.metrics_on_split["use_cp_upper"], policy.metrics_on_split["cp_alpha"]) == (True, 0.1)


def test_the_fingerprint_does_not_cover_the_metrics(ws, tmp_path, capsys) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws["records"], "--error-budget", "0.1", "-o", str(out)]) == 0
    (policy,) = load_policies(out)
    assert policy.metrics_on_split
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["policies"][0]["metrics_on_split"] = {}
    out.write_text(json.dumps(doc), encoding="utf-8")
    (without,) = load_policies(out)
    assert without.policy_sha256 == policy.policy_sha256


@pytest.mark.parametrize("policy_key", ["policy", "policy_categorical"])
def test_the_demo_writes_metrics_for_every_policy(ws, policy_key) -> None:
    policies = load_policies(ws[policy_key])
    assert policies
    for policy in policies:
        metrics = policy.metrics_on_split
        assert metrics["selection"] == policy.mode and metrics["n"] == policy.n
        assert 0 <= metrics["errors"] <= metrics["accepted"] <= metrics["n"]


def test_a_false_pass_policy_keeps_its_own_shape(ws, tmp_path, capsys) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws["records"], "--max-false-pass", "0.05", "-o", str(out)]) == 0
    (policy,) = load_policies(out)
    assert policy.metrics_on_split["selection"] == "max_false_pass" and "accepted" not in policy.metrics_on_split
