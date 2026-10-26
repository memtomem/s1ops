"""The demo's choice and score folds (#16): artifacts the README's commands reproduce, routed like
the gate, generated without touching any noul file."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import read_records, select
from s1ops.contracts.schemas import load_policies
from s1ops.demo import _categorical_rows, generate_workspace
from s1ops.eval.metrics import top_label
from s1ops.improve.calibrate import load_fits, sha256_file
from s1ops.lint.lint import lint_path
from s1ops.runtime.gate import DecisionGate
from test_demo import _load, _strip_clock, _workspace

CHOICE, SCORE = "q-a70bd97f0b7f1055", "q-5c05e56162b3eaa0"
README = Path(__file__).resolve().parents[1] / "README.md"

# Every noul-era file as main @ 04e23a8 generated it, clock keys stripped.
MAIN_DIGESTS = {
    "features.jsonl": "51c3a928536f1f8d8a18d87fc459daec04e6bfa8dea3d66a56cdaf4608435510",
    "fits.json": "f62a5a22d82f776b8cd571a0782a35050e09ae5ee3ab33e846841d14963fb129",
    "judge_grades.jsonl": "6789b7f00c88b9cdf7bce131959ec7b39fe9cc483541ad001d54a33e2dc9abcf",
    "policy.json": "d3da5cd88f81bdd93057fe51af61af047bfde487914f1d85c8929a287bf64d93",
    "probe_records.jsonl": "b1d58b94e592473634675a37d921edb19555eb9a9d2953e1b9fb25edc9a9ff47",
    "questions.json": "25436640f677c4d79b16805c053d9a75c4ad118a4d5c9f382009df650eba7cfe",
    "records.jsonl": "780927ee3d457ab0cc1f0abd65498c7f569e7b24a49e5187f25ebe51e6ab46da",
    "records_canary.jsonl": "c6c48a5432dfac4b6ea25ec8d107183b866db0a69df178d5f60ac3669eb1d582",
    "records_kev.jsonl": "82c43a365f05ea7da8675f966a302a078729d0a5eeaa7d6a3da142ef48bf6198",
    "records_train.jsonl": "4d45fd1f3e2daaaa0d5cf18ea743c0db9052607f4f7cb643079e4ae5576ac082",
    "states_local.jsonl": "37052540734f2edb2b615662f2c3fd560f68e5bf1cc013deed5645f9fb6a3b8e",
}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def test_the_noul_files_are_unchanged(ws) -> None:
    files = _workspace(Path(ws["records"]).parent)
    urgency = files["questions.json"].pop("urgency")  # the one addition to questions.json
    assert (urgency["type"], len(urgency["criteria"])) == ("score", 4)
    assert {name: _digest(files[name]) for name in MAIN_DIGESTS} == MAIN_DIGESTS


def test_the_folds_have_the_documented_shape(ws) -> None:
    for name in ("records_categorical_train", "records_categorical"):
        rows = _load(Path(ws[name]))
        assert Counter((r["question_id"], r["primitive"]) for r in rows) == {(CHOICE, "choice"): 150, (SCORE, "score"): 150}
        read_records(ws[name], use_calib=name == "records_categorical")  # every row parses
        for r in rows:  # the answer fields agree with the distribution
            probs = r["probabilities"]
            assert r["confidence"] == max(probs.values())
            if r["primitive"] == "choice":
                assert probs[r["choice"]] == r["confidence"]
            else:
                assert r["score"] == round(sum(int(k) * p for k, p in probs.items()), 6)
    temperatures = [f.params["temperature"] for f in load_fits(ws["fits_categorical"])]
    assert len(temperatures) == 2 and all(t > 1 for t in temperatures)  # overconfident, as generated
    policies = {p.question_id: p for p in load_policies(ws["policy_categorical"])}
    assert set(policies) == {CHOICE, SCORE}
    for grp in select(read_records(ws["records_categorical"], use_calib=True)).groups:
        pol = policies[grp.question_id]
        assert pol.calibrated and pol.mode == "error_budget" and pol.threshold is not None
        conf, _ = top_label(grp.P, grp.y)
        assert 0 < (conf >= pol.threshold).mean() < 1  # some answers are accepted, some escalated


def test_the_readme_commands_reproduce_the_artifacts(ws, tmp_path: Path, capsys) -> None:
    d = Path(ws["records"]).parent
    fits, applied, policy = tmp_path / "fits.json", tmp_path / "applied.jsonl", tmp_path / "policy.json"
    assert cli_main(["fit", ws["records_categorical_train"], "-o", str(fits), "--method", "temperature"]) == 0
    assert cli_main(["apply", ws["records_categorical"], "--fit", ws["fits_categorical"], "-o", str(applied)]) == 0
    assert cli_main(["policy", "tune", ws["records_categorical"], "-o", str(policy), "--error-budget", "0.10", "--use-calib"]) == 0
    capsys.readouterr()
    assert _strip_clock(_load(fits)) == _strip_clock(_load(d / "fits_categorical.json"))
    assert _strip_clock(_load(applied)) == _strip_clock(_load(d / "records_categorical.jsonl"))
    # The demo leaves out source_sha256, as its noul policy does: the tuned file embeds fitted_at.
    tuned, demo = _strip_clock(_load(policy)), _strip_clock(_load(d / "policy_categorical.json"))
    for p in tuned["policies"]:
        assert p.pop("source_sha256") == sha256_file(ws["records_categorical"])
    for p in demo["policies"]:
        assert p.pop("source_sha256") == ""
    assert tuned == demo


@pytest.mark.parametrize(
    "question, probs, action",
    [
        (CHOICE, '{"billing": 0.97, "technical": 0.02, "account": 0.01}', "SELECT:billing"),
        (SCORE, '{"0": 0.1, "1": 0.45, "2": 0.4, "3": 0.05}', "ESCALATE"),
    ],
)
def test_the_readme_gate_examples(ws, question: str, probs: str, action: str, capsys) -> None:
    readme = README.read_text()
    assert f"--question {question} --model jev-1.13.0" in readme and f"--probs '{probs}'   # {action}" in readme
    args = ["gate", ws["policy_categorical"], "--fit", ws["fits_categorical"], "--question", question, "--model", "jev-1.13.0", "--probs", probs]
    assert cli_main(args) == 0
    assert json.loads(capsys.readouterr().out)["action"] == action


def test_exports_route_every_held_out_record_like_the_gate(ws, tmp_path: Path, capsys) -> None:
    gate = DecisionGate.from_files(ws["policy_categorical"], ws["fits_categorical"])
    expected: dict[str, str] = {}
    for rec in read_records(ws["records_categorical"]):
        fn = gate.evaluate_choice if rec.primitive == "choice" else gate.evaluate_score
        expected[rec.item_id] = fn(question_id=rec.question_id, model_served=rec.model_served, probs_raw=rec.raw["probabilities"]).action
    assert len(expected) == 300
    assert {(item.split("-")[2], action == "ESCALATE") for item, action in expected.items()} == {
        (p, e) for p in ("choice", "score") for e in (False, True)
    }
    phoenix, langfuse = tmp_path / "spans.jsonl", tmp_path / "langfuse.json"
    policy = ["--policy", ws["policy_categorical"]]
    assert cli_main(["export", "phoenix", ws["records_categorical"], "-o", str(phoenix), *policy]) == 0
    assert cli_main(["export", "langfuse", ws["records_categorical"], "-o", str(langfuse), *policy]) == 0
    capsys.readouterr()
    spans = [json.loads(line)["attributes"] for line in phoenix.read_text().splitlines()]
    assert {a["s1.item_id"]: a["s1.policy_action"] for a in spans} == expected
    observations = json.loads(langfuse.read_text())["observations"]
    assert {o["input"]["item_id"]: o["output"]["policy_action"] for o in observations} == expected


def test_the_score_question_lints_clean_and_is_probed(ws, tmp_path: Path, capsys) -> None:
    only = tmp_path / "urgency.json"
    only.write_text(json.dumps({"urgency": json.loads(Path(ws["questions"]).read_text())["urgency"]}))
    assert lint_path(only).findings == []
    out = tmp_path / "probe.jsonl"
    args = ["probe", ws["questions"], "--states", ws["states_local"], "-o", str(out), "--ledger", str(tmp_path / "ledger.jsonl"), "--cap-usd", "0.50"]
    assert cli_main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_failed"] == 0 and summary["spent_usd"] < 0.50
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert sum(r["question_id"] == SCORE for r in rows) == 160


class _TieAfterRounding:
    """Draws that give two options 0.4336 each: rounded, the three sum to 1.001, and taking the
    residual off the first option hands the top to the second."""

    def __init__(self) -> None:
        self._gauss = iter([0.0, 2.6, (4.16 + math.log(0.30627)) / 1.6])

    def randrange(self, k: int) -> int:
        return 0

    def random(self) -> float:
        return 0.0

    def choice(self, xs: list):
        return xs[0]

    def gauss(self, mu: float, sigma: float) -> float:
        return next(self._gauss)

    def uniform(self, a: float, b: float) -> float:
        return a


def test_the_answer_follows_the_rounded_distribution() -> None:
    (row,) = _categorical_rows(_TieAfterRounding(), "q", "choice", ["a", "b", "c"], "d", "eval", 1)
    assert row["probabilities"] == {"a": 0.433, "b": 0.434, "c": 0.133}
    assert (row["choice"], row["confidence"]) == ("b", 0.434)
