"""`--fit` on `s1 drift` and `s1 export` is `s1 apply --pass-through-unfitted` in memory (#21).

Every parity test runs the command both ways: with `--fit`, and through a file written by `s1 apply`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import s1ops
from s1ops.cli import main as cli_main
from s1ops.contracts.records import parse_record, read_records, reparse
from s1ops.demo import generate_workspace
from s1ops.improve.calibrate import FitApplyError, FitError, apply_fits, load_fits

FIT_MODEL, NEW_MODEL = "jev-1.13.0", "jev-1.14.0-rc1"


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return generate_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _stripped(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "calib"} for r in rows]


def _failed_row(like: dict) -> dict:
    return {k: v for k, v in like.items() if k != "p"} | {"model_served": None, "error_code": "timeout", "item_id": "failed"}


def _live_noul(ws: dict, tmp_path: Path, *, only_fitted: bool = False) -> tuple[str, str]:
    """The demo's reference and canary as a live run has them: no `calib`, one failed record."""
    ref = _stripped(_rows(ws["records"]))
    cur = _stripped(_rows(ws["records_canary"]))
    if only_fitted:
        cur = [r for r in cur if r["model_served"] == FIT_MODEL]
    cur.insert(1, _failed_row(cur[0]))
    return _write(tmp_path / "live_ref.jsonl", ref), _write(tmp_path / "live_cur.jsonl", cur)


def _live_categorical(ws: dict, tmp_path: Path) -> tuple[str, str]:
    """Categorical records whose first choice row is tied at the top with its keys out of order."""
    rows = _stripped(_rows(ws["records_categorical"]))
    n = next(i for i, r in enumerate(rows) if r["primitive"] == "choice")
    first, second, *rest = sorted(rows[n]["probabilities"], reverse=True)
    share = 0.8 / 2
    tied = {first: share, second: share}
    tied.update({k: 0.2 / len(rest) for k in rest})
    rows[n] = {**rows[n], "probabilities": tied, "choice": first, "confidence": share}
    rows[n].pop("label", None)
    return _write(tmp_path / "cat_ref.jsonl", rows), _write(tmp_path / "cat_cur.jsonl", rows)


def _applied_file(records: str, fits: str, out: Path, *flags: str) -> str:
    assert cli_main(["apply", records, "--fit", fits, "-o", str(out), "--pass-through-unfitted", *flags]) == 0
    return str(out)


def _run_json(capsys, argv: list[str]) -> dict:
    capsys.readouterr()
    assert cli_main(argv) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("workload, flags", [
    ("noul", ["--policy"]),
    ("noul-fitted", ["--calibrated"]),
    ("noul-fitted", ["--calibrated", "--policy"]),
    ("categorical", ["--policy"]),
    ("categorical", ["--calibrated"]),
])
def test_drift_report_equals_the_one_over_applied_files(ws, tmp_path: Path, capsys, workload: str, flags: list[str]) -> None:
    if workload == "categorical":
        ref, cur = _live_categorical(ws, tmp_path)
        fits, policy = ws["fits_categorical"], ws["policy_categorical"]
    else:
        ref, cur = _live_noul(ws, tmp_path, only_fitted=workload == "noul-fitted")
        fits, policy = ws["fits"], ws["policy"]
    extra = [policy if f == "--policy" else None for f in flags]
    opts = [x for f, p in zip(flags, extra) for x in (f, p) if x is not None]

    in_memory = _run_json(capsys, ["drift", ref, cur, "--fit", fits, *opts])
    applied_ref = _applied_file(ref, fits, tmp_path / "ar.jsonl")
    applied_cur = _applied_file(cur, fits, tmp_path / "ac.jsonl")
    via_files = _run_json(capsys, ["drift", applied_ref, applied_cur, *opts])

    assert set(in_memory) - set(via_files) == {"fit"}
    in_memory.pop("fit")
    assert in_memory == via_files
    if workload != "categorical":
        # Both routes share `apply_fits`, so parity alone would not see a row it lost.
        (q,) = in_memory["questions"]
        n_answered = 80 if workload == "noul-fitted" else 120
        assert (q["n_cur"], q["n_cur_answered"]) == (n_answered + 1, n_answered)
        assert q["failure_rate_cur"] == round(1 / (n_answered + 1), 6)
    if "--policy" in flags:
        assert via_files["questions"][0]["escalation_rate_cur"] is not None
    if "--calibrated" in flags:
        assert via_files["distribution"] == "calibrated"


@pytest.mark.parametrize("target", ["langfuse", "phoenix"])
@pytest.mark.parametrize("workload", ["noul", "categorical"])
@pytest.mark.parametrize("with_policy", [True, False])
def test_export_files_equal_the_ones_from_applied_files(ws, tmp_path: Path, capsys, target: str, workload: str, with_policy: bool) -> None:
    if workload == "categorical":
        _, live = _live_categorical(ws, tmp_path)
        fits, policy = ws["fits_categorical"], ws["policy_categorical"]
    else:
        _, live = _live_noul(ws, tmp_path)
        fits, policy = ws["fits"], ws["policy"]
    pol = ["--policy", policy] if with_policy else []
    in_memory, via_file = tmp_path / "mem", tmp_path / "file"

    assert cli_main(["export", target, live, "-o", str(in_memory), "--fit", fits, *pol]) == 0
    applied = _applied_file(live, fits, tmp_path / "a.jsonl")
    assert cli_main(["export", target, applied, "-o", str(via_file), *pol]) == 0

    assert in_memory.read_bytes() == via_file.read_bytes()


def test_a_tied_choice_exports_the_answer_of_the_written_key_order(ws, tmp_path: Path, capsys) -> None:
    _, live = _live_categorical(ws, tmp_path)
    tied = next(r for r in _rows(live) if r["primitive"] == "choice")
    assert list(tied["probabilities"]) != sorted(tied["probabilities"])
    out = tmp_path / "ph.jsonl"
    assert cli_main(["export", "phoenix", live, "-o", str(out), "--fit", ws["fits_categorical"], "--policy", ws["policy_categorical"]]) == 0
    span = next(s for s in _rows(out) if s["attributes"]["s1.item_id"] == tied["item_id"] and s["attributes"]["s1.primitive"] == "choice")
    top = max(tied["probabilities"].values())
    level = [k for k, q in tied["probabilities"].items() if q == top]
    assert len(level) == 2 and level != sorted(level)
    assert span["attributes"]["s1.predicted"] == sorted(level)[0]
    assert span["attributes"]["s1.policy_action"] == "ESCALATE"


def test_records_are_bit_identical_to_the_written_file(ws, tmp_path: Path, capsys) -> None:
    for live, fits in (
        (_live_noul(ws, tmp_path)[1], ws["fits"]),
        (_live_categorical(ws, tmp_path)[1], ws["fits_categorical"]),
    ):
        applied = apply_fits(read_records(live), load_fits(fits), pass_through_unfitted=True)
        from_file = read_records(_applied_file(live, fits, tmp_path / "w.jsonl"))
        assert len(applied.records) == len(from_file) > 0
        for mine, theirs in zip(applied.records, from_file):
            assert mine.raw == theirs.raw
            assert list(mine.raw) == list(theirs.raw)
            assert mine.classes == theirs.classes
            assert [q.hex() for q in mine.probs] == [q.hex() for q in theirs.probs]
            if mine.model_served is not None:
                ours, written = reparse(mine, use_calib=True), reparse(theirs, use_calib=True)
                assert [q.hex() for q in ours.probs] == [q.hex() for q in written.probs]


def test_the_report_says_what_was_calibrated(ws, tmp_path: Path, capsys) -> None:
    ref, cur = _live_noul(ws, tmp_path)
    rep = _run_json(capsys, ["drift", ref, cur, "--fit", ws["fits"], "--policy", ws["policy"]])
    (fit,) = load_fits(ws["fits"])
    assert rep["fit"] == {
        "reference": {"calibrated": 160, "passed_through": []},
        "current": {
            "calibrated": 80,
            "passed_through": [{
                "question_id": fit.question_id, "model_served": NEW_MODEL, "primitive": "noul", "n": 40,
                "reason": f"fit is for model_served {FIT_MODEL}; this answer was served by {NEW_MODEL}",
            }],
        },
    }
    assert "fit" not in _run_json(capsys, ["drift", ws["records"], ws["records_canary"], "--policy", ws["policy"]])


@pytest.mark.parametrize("target", ["langfuse", "phoenix"])
def test_the_printed_export_summary_says_it_and_the_file_does_not(ws, tmp_path: Path, capsys, target: str) -> None:
    _, live = _live_noul(ws, tmp_path)
    out = tmp_path / "o"
    printed = _run_json(capsys, ["export", target, live, "-o", str(out), "--fit", ws["fits"], "--policy", ws["policy"]])
    assert printed["fit"]["calibrated"] == 80
    assert [g["n"] for g in printed["fit"]["passed_through"]] == [40]
    assert '"passed_through"' not in out.read_text()


def test_without_fit_a_live_canary_is_still_refused_and_the_hint_names_it(ws, tmp_path: Path, capsys) -> None:
    ref, cur = _live_noul(ws, tmp_path)
    assert cli_main(["drift", ref, cur, "--policy", ws["policy"]]) == 1
    assert "pass `s1 drift` or `s1 export` a `--fit` with a fit for this group" in capsys.readouterr().err


def test_a_fit_file_without_this_group_leaves_the_policy_refusal_standing(ws, tmp_path: Path, capsys) -> None:
    ref, cur = _live_noul(ws, tmp_path)
    doc = json.loads(Path(ws["fits"]).read_text())
    doc["fits"][0]["question_id"] = "q-other"
    other = tmp_path / "other_fits.json"
    other.write_text(json.dumps(doc))
    assert cli_main(["drift", ref, cur, "--policy", ws["policy"], "--fit", str(other)]) == 1
    assert "this record has none" in capsys.readouterr().err


def test_calibrated_comparison_refuses_a_group_the_fit_file_does_not_cover(ws, tmp_path: Path, capsys) -> None:
    ref, cur = _live_noul(ws, tmp_path)
    assert cli_main(["drift", ref, cur, "--calibrated", "--fit", ws["fits"]]) == 1
    err = capsys.readouterr().err
    assert "current records: line 1:" in err
    assert "pass a `--fit` with a fit for this record's group" in err


def _other_fit(ws: dict, tmp_path: Path) -> str:
    doc = json.loads(Path(ws["fits"]).read_text())
    doc["fits"][0]["params"]["temperature"] *= 1.5
    path = tmp_path / "other_fits.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_a_policy_tuned_through_another_fit_is_still_refused(ws, tmp_path: Path, capsys) -> None:
    ref, cur = _live_noul(ws, tmp_path)
    assert cli_main(["drift", ref, cur, "--policy", ws["policy"], "--fit", _other_fit(ws, tmp_path)]) == 1
    assert "was tuned through fit" in capsys.readouterr().err


def test_the_fit_replaces_a_stored_calib_and_a_passed_through_record_keeps_its_own(ws, tmp_path: Path, capsys) -> None:
    (fit,) = load_fits(ws["fits"])
    rows = _rows(ws["records_canary"])
    fitted = next(i for i, r in enumerate(rows) if r["model_served"] == FIT_MODEL)
    new = next(i for i, r in enumerate(rows) if r["model_served"] == NEW_MODEL)
    rows[fitted]["calib"] = {"method": "temperature", "fit_split": "cal", "fit_n": 9, "fitted_at": "t", "p": 0.25, "fit_sha256": "stale"}
    kept = {"method": "temperature", "fit_split": "cal", "fit_n": 9, "fitted_at": "t", "p": 0.25, "fit_sha256": "its-own"}
    rows[new]["calib"] = kept
    live = _write(tmp_path / "r.jsonl", rows)

    applied = apply_fits(read_records(live), load_fits(ws["fits"]), pass_through_unfitted=True).records
    assert applied[fitted].raw["calib"]["fit_sha256"] == fit.fit_sha256
    assert applied[new].raw["calib"] == kept

    # The stale `calib` above would refuse under the policy; with --fit it is replaced.
    assert cli_main(["drift", live, live, "--policy", ws["policy"]]) == 1
    capsys.readouterr()
    assert cli_main(["drift", live, live, "--policy", ws["policy"], "--fit", ws["fits"]]) == 0


def test_a_malformed_stored_calib_is_replaced_when_fitted_and_refused_when_not(ws, tmp_path: Path, capsys) -> None:
    rows = _stripped(_rows(ws["records_canary"]))
    fitted = next(i for i, r in enumerate(rows) if r["model_served"] == FIT_MODEL)
    new = next(i for i, r in enumerate(rows) if r["model_served"] == NEW_MODEL)

    bad_fitted = [dict(r) for r in rows]
    bad_fitted[fitted]["calib"] = {"p": "bad"}
    live = _write(tmp_path / "f.jsonl", bad_fitted)
    assert cli_main(["drift", live, live, "--policy", ws["policy"], "--fit", ws["fits"]]) == 0
    capsys.readouterr()

    bad_new = [dict(r) for r in rows]
    bad_new[new]["calib"] = {"p": "bad"}
    live = _write(tmp_path / "n.jsonl", bad_new)
    assert cli_main(["drift", live, live, "--policy", ws["policy"], "--fit", ws["fits"]]) == 1
    assert f"{live}: line {new + 1}: calib.p" in capsys.readouterr().err
    assert cli_main(["export", "phoenix", live, "-o", str(tmp_path / "o"), "--fit", ws["fits"]]) == 1
    assert f"{live}: line {new + 1}: calib.p" in capsys.readouterr().err


@pytest.mark.parametrize("side", ["reference", "current"])
def test_in_sample_items_exit_2_naming_the_file_and_are_allowed_on_request(ws, tmp_path: Path, capsys, side: str) -> None:
    held_out = _write(tmp_path / "h.jsonl", _stripped(_rows(ws["records"])))
    train = ws["records_train"]
    ref, cur = (train, held_out) if side == "reference" else (held_out, train)
    common = ["drift", ref, cur, "--policy", ws["policy"], "--fit", ws["fits"]]

    assert cli_main(common) == 2
    assert capsys.readouterr().err.startswith(f"s1: {train}: line 1: refusing in-sample calibration apply on ")

    in_memory = _run_json(capsys, [*common, "--allow-in-sample"])
    a_ref = _applied_file(ref, ws["fits"], tmp_path / "ar.jsonl", "--allow-in-sample")
    a_cur = _applied_file(cur, ws["fits"], tmp_path / "ac.jsonl", "--allow-in-sample")
    via_files = _run_json(capsys, ["drift", a_ref, a_cur, "--policy", ws["policy"]])
    in_memory.pop("fit")
    assert in_memory == via_files


@pytest.mark.parametrize("target", ["langfuse", "phoenix"])
def test_export_refuses_in_sample_items_too(ws, tmp_path: Path, capsys, target: str) -> None:
    out = tmp_path / "o"
    assert cli_main(["export", target, ws["records_train"], "-o", str(out), "--fit", ws["fits"]]) == 2
    assert not out.exists()
    assert "--allow-in-sample" in capsys.readouterr().err
    assert cli_main(["export", target, ws["records_train"], "-o", str(out), "--fit", ws["fits"], "--allow-in-sample"]) == 0


def _class_mismatch(ws: dict) -> list[dict]:
    rows = _stripped(_rows(ws["records_categorical"]))
    n = next(i for i, r in enumerate(rows) if r["primitive"] == "choice")
    probs = rows[n]["probabilities"]
    first, *rest = probs
    rows[n] = {**rows[n], "probabilities": {"not-an-option": probs[first], **{k: probs[k] for k in rest}}}
    rows[n]["choice"] = max(rows[n]["probabilities"], key=rows[n]["probabilities"].get)
    rows[n].pop("label", None)
    return rows


@pytest.mark.parametrize("side", ["reference", "current"])
def test_a_class_mismatch_exits_1_naming_the_file_and_its_original_line(ws, tmp_path: Path, capsys, side: str) -> None:
    rows = _class_mismatch(ws)
    n = next(i for i, r in enumerate(rows) if "not-an-option" in r.get("probabilities", {}))
    bad = tmp_path / "bad.jsonl"
    # A blank line above it: `s1 apply` would compact it away, and the error names the input's line.
    bad.write_text("\n" + "".join(json.dumps(r) + "\n" for r in rows))
    good = _write(tmp_path / "good.jsonl", [r for r in rows if "not-an-option" not in r.get("probabilities", {})])
    ref, cur = (str(bad), good) if side == "reference" else (good, str(bad))
    assert cli_main(["drift", ref, cur, "--calibrated", "--fit", ws["fits_categorical"]]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"s1: {bad}: line {n + 2}: refusing to apply fit: answer classes ")
    assert err.count("\n") == 1


def test_an_unreadable_row_is_refused_as_the_applied_file_would_be(ws, tmp_path: Path, capsys) -> None:
    rows = _stripped(_rows(ws["records_canary"]))
    path = tmp_path / "over.jsonl"
    lines = [json.dumps(r) for r in rows]
    lines[2] = lines[2][:-1] + ', "extra": 1e400}'
    path.write_text("\n\n".join(lines) + "\n")
    assert cli_main(["export", "phoenix", str(path), "-o", str(tmp_path / "o"), "--fit", ws["fits"]]) == 1
    assert capsys.readouterr().err.startswith(f"s1: {path}: line 5: not valid JSON: non-finite number Infinity")


@pytest.mark.parametrize("argv, message", [
    (["drift", "{r}", "{r}", "--allow-in-sample"], "--allow-in-sample applies only with --fit"),
    (["export", "phoenix", "{r}", "-o", "{o}", "--allow-in-sample"], "--allow-in-sample applies only with --fit"),
    (["export", "langfuse", "{r}", "-o", "{o}", "--allow-in-sample"], "--allow-in-sample applies only with --fit"),
    (["drift", "{r}", "{r}", "--fit", "{f}"], "--fit does not change a raw comparison without a policy"),
    (["drift", "{r}", "{r}", "--raw-only", "--fit", "{f}"], "--fit does not change a raw comparison without a policy"),
])
def test_flag_misuse_exits_1(ws, tmp_path: Path, capsys, argv: list[str], message: str) -> None:
    subs = {"{r}": ws["records"], "{o}": str(tmp_path / "o"), "{f}": ws["fits"]}
    assert cli_main([subs.get(a, a) for a in argv]) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_raw_only_with_a_policy_and_a_fit_is_valid(ws, tmp_path: Path, capsys) -> None:
    ref, cur = _live_noul(ws, tmp_path)
    rep = _run_json(capsys, ["drift", ref, cur, "--raw-only", "--policy", ws["policy"], "--fit", ws["fits"]])
    assert rep["distribution"] == "raw"
    assert rep["questions"][0]["escalation_rate_cur"] is not None


def test_a_missing_fit_file_names_its_producer(ws, tmp_path: Path, capsys) -> None:
    assert cli_main(["drift", ws["records"], ws["records"], "--policy", ws["policy"], "--fit", str(tmp_path / "nope.json")]) == 1
    assert "(run `s1 fit` first)" in capsys.readouterr().err


def test_state_reaches_neither_report_nor_export(ws, tmp_path: Path, capsys) -> None:
    secret = "SECRET-ITEM-TEXT"
    rows = [{**r, "state": secret} for r in _stripped(_rows(ws["records_canary"]))]
    live = _write(tmp_path / "s.jsonl", rows)
    capsys.readouterr()
    assert cli_main(["drift", live, live, "--policy", ws["policy"], "--fit", ws["fits"]]) == 0
    assert secret not in capsys.readouterr().out
    for target in ("langfuse", "phoenix"):
        out = tmp_path / target
        assert cli_main(["export", target, live, "-o", str(out), "--policy", ws["policy"], "--fit", ws["fits"]]) == 0
        assert secret not in out.read_text()
    applied = apply_fits(read_records(live), load_fits(ws["fits"]), pass_through_unfitted=True)
    assert all("state" not in r.raw for r in applied.records)
    assert secret not in "".join(applied.lines)


# `model_mismatch` comes from the fit lookup: a word in an id or a path decides nothing.
def _renamed_fit(ws: dict, tmp_path: Path, **fields: str) -> str:
    doc = json.loads(Path(ws["fits"]).read_text())
    doc["fits"][0].update(fields)
    path = tmp_path / "renamed_fits.json"
    path.write_text(json.dumps(doc))
    return str(path)


def _noul_row(ws: dict, **fields: object) -> dict:
    row = _stripped(_rows(ws["records_canary"]))[0]
    return {**row, **fields}


def test_a_question_named_after_the_word_exits_1_when_it_has_no_fit(ws, tmp_path: Path, capsys) -> None:
    path = _write(tmp_path / "r.jsonl", [_noul_row(ws, question_id="q-model_served")])
    assert cli_main(["apply", path, "--fit", ws["fits"], "-o", str(tmp_path / "o")]) == 1
    assert "no fit for question_id q-model_served" in capsys.readouterr().err


def test_a_known_question_under_another_primitive_exits_1_whatever_its_name(ws, tmp_path: Path, capsys) -> None:
    fits = _renamed_fit(ws, tmp_path, question_id="q-model_served")
    row = {k: v for k, v in _noul_row(ws, question_id="q-model_served").items() if k != "p"}
    row.update({"primitive": "choice", "choice": "A", "confidence": 0.7, "probabilities": {"A": 0.7, "B": 0.3}})
    path = _write(tmp_path / "r.jsonl", [row])
    assert cli_main(["apply", path, "--fit", fits, "-o", str(tmp_path / "o")]) == 1
    assert "is for primitive noul, not choice" in capsys.readouterr().err


def test_a_path_with_the_word_does_not_make_a_mismatch(ws, tmp_path: Path, capsys) -> None:
    folder = tmp_path / "model_served"
    folder.mkdir()
    path = _write(folder / "r.jsonl", [_noul_row(ws, question_id="q-unknown")])
    assert cli_main(["apply", path, "--fit", ws["fits"], "-o", str(tmp_path / "o")]) == 1
    capsys.readouterr()


def test_another_model_of_a_fitted_question_exits_3(ws, tmp_path: Path, capsys) -> None:
    rows = [r for r in _stripped(_rows(ws["records_canary"])) if r["model_served"] == NEW_MODEL]
    path = _write(tmp_path / "r.jsonl", rows)
    assert cli_main(["apply", path, "--fit", ws["fits"], "-o", str(tmp_path / "o")]) == 3
    assert capsys.readouterr().err.startswith(f"s1: {path}: line 1: refusing to apply fit: fit is for model_served ")


@pytest.mark.parametrize("kind, code", [("in_sample", 2), ("model_mismatch", 3), ("other", 1)])
def test_each_kind_maps_to_its_exit_and_debug_reraises(ws, tmp_path: Path, capsys, monkeypatch, kind: str, code: int) -> None:
    if kind == "in_sample":
        path = ws["records_train"]
    elif kind == "model_mismatch":
        path = _write(tmp_path / "r.jsonl", [r for r in _stripped(_rows(ws["records_canary"])) if r["model_served"] == NEW_MODEL])
    else:
        path = _write(tmp_path / "r.jsonl", [_noul_row(ws, question_id="q-unknown")])
    argv = ["apply", path, "--fit", ws["fits"], "-o", str(tmp_path / "o")]
    assert cli_main(argv) == code
    assert capsys.readouterr().err.count("\n") == 1
    monkeypatch.setenv("S1_DEBUG", "1")
    with pytest.raises(FitApplyError) as e:
        cli_main(argv)
    assert e.value.kind == kind
    assert e.value.line == 1
    assert e.value.path == path


def test_the_result_does_not_depend_on_how_the_caller_parsed(ws, tmp_path: Path) -> None:
    fits = load_fits(ws["fits"])
    # A stored `calib` that differs from what the fit makes: it is replaced, and a calibrated parse
    # must not be transformed again.
    rows = _rows(ws["records_canary"])
    fitted = next(i for i, r in enumerate(rows) if r["model_served"] == FIT_MODEL)
    stale = {"method": "temperature", "fit_split": "cal", "fit_n": 9, "fitted_at": "t", "p": 0.01, "fit_sha256": "stale"}
    expected = apply_fits([parse_record(_stripped(rows)[fitted])], fits).lines
    rows[fitted]["calib"] = stale
    rows.insert(fitted + 1, _failed_row(_stripped(rows)[fitted]))
    path = _write(tmp_path / "r.jsonl", rows)
    raw = apply_fits(read_records(path), fits, pass_through_unfitted=True).lines
    assert raw[fitted] == expected[0]
    assert json.loads(raw[fitted])["calib"]["p"] != stale["p"]
    for kwargs in ({"use_calib": True}, {"require_answer": False}):
        assert apply_fits(read_records(path, **kwargs), fits, pass_through_unfitted=True).lines == raw


def test_a_row_only_a_looser_tolerance_accepts_is_refused(ws) -> None:
    row = _noul_row(ws)
    row = {k: v for k, v in row.items() if k != "p"}
    row.update({"primitive": "choice", "choice": "A", "confidence": 0.7, "probabilities": {"A": 0.7, "B": 0.35}})
    loose = parse_record(row, sum_tol=0.1)
    with pytest.raises(ValueError, match="sum to 1.05"):
        apply_fits([loose], load_fits(ws["fits"]), pass_through_unfitted=True)


def test_without_pass_through_a_record_with_no_fit_raises(ws) -> None:
    rec = parse_record(_noul_row(ws, question_id="q-unknown"))
    with pytest.raises(FitApplyError) as e:
        apply_fits([rec], load_fits(ws["fits"]))
    assert (e.value.kind, e.value.path) == ("other", None)
    assert str(e.value).startswith("line 1: refusing to apply fit:")


def test_the_sdk_names_are_public(ws) -> None:
    assert s1ops.apply_fits is apply_fits
    assert s1ops.FitApplyError is FitApplyError
    assert {"apply_fits", "AppliedFits", "FitApplyError"} <= set(s1ops.__all__)


def test_a_repeated_group_key_is_refused_before_any_record(ws) -> None:
    # `fit_for` takes the first fit of a group and `fitted_items` the last one's items, so a repeat
    # would calibrate an item the first fit was trained on without calling it in-sample.
    (fit,) = load_fits(ws["fits"])
    dataset, item_id = fit.items[0]
    row = _noul_row(ws, dataset=dataset, item_id=item_id, model_served=FIT_MODEL)
    trained_on_other_items = replace(fit, items=[["other", "item"]])
    consumed = []

    def records():
        consumed.append(True)
        yield parse_record(row)

    with pytest.raises(FitError, match="more than one fit for"):
        apply_fits(records(), [fit, trained_on_other_items])
    assert consumed == []
