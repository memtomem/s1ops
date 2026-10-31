"""Row schemas for `features.jsonl` and probe records: a malformed row is one error line, not a traceback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace
from s1ops.improve.slices import parse_feature_rows
from s1ops.lint.lint import ProbeInputError, audit_probe_records
from s1ops.web.app import create_app


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, str]:
    return generate_workspace(tmp_path / "ws")


def _error_line(capsys: pytest.CaptureFixture[str]) -> str:
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("s1: ") and err.count("\n") == 1
    assert "Traceback" not in err
    return err


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path: str | Path, rows: list[dict]) -> str:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _client(ws: dict[str, str]) -> TestClient:
    return TestClient(create_app(workspace_dir=Path(ws["records"]).parent))


_ROW = {"dataset": "ds", "item_id": "a", "slice": "s", "features": {"lang": "en", "has_code": True}}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dataset": []}, "row 1: 'dataset' must be a non-empty string or null, got list"),
        ({"dataset": ""}, "row 1: 'dataset' must be a non-empty string or null, got str"),
        ({"item_id": True}, "row 1: 'item_id' must be a non-empty string or an integer, got bool"),
        ({"item_id": ""}, "row 1: 'item_id' must be a non-empty string or an integer, got str"),
        ({"item_id": 1.5}, "row 1: 'item_id' must be a non-empty string or an integer, got float"),
        ({"slice": 3}, "row 1: 'slice' must be a string, got int"),
        ({"features": ["x"]}, "row 1: 'features' must be an object, got list"),
        ({"features": {"tags": ["a"]}}, "row 1: each 'features' value must be a string, number, boolean or null, got list"),
        ({"features": {"slice": 1}}, "row 1: 'features.slice' must be a string, got int"),
    ],
)
def test_feature_row_schema(change: dict, message: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_feature_rows([{**_ROW, **change}])
    assert str(e.value) == message


def test_feature_rows_missing_item_and_duplicates() -> None:
    with pytest.raises(ValueError) as e:
        parse_feature_rows([{"dataset": "ds"}])
    assert str(e.value) == "row 1 is missing 'item_id'"
    with pytest.raises(ValueError) as e:
        parse_feature_rows([_ROW, {**_ROW, "slice": "t"}])
    assert str(e.value) == "row 2 is the same item as row 1"
    # An integer id and its string are the same item, as records key it.
    with pytest.raises(ValueError, match="row 2 is the same item as row 1"):
        parse_feature_rows([{**_ROW, "item_id": 7}, {**_ROW, "item_id": "7"}])


def test_feature_rows_index_as_records_key_items() -> None:
    features, slices = parse_feature_rows(
        [
            {"item_id": 7, "features": {"slice": "f", "n": 2, "x": None}},
            {"dataset": "ds", "item_id": "b", "slice": "row", "features": {"slice": "feat"}},
        ]
    )
    assert features == {(None, "7"): {"slice": "f", "n": 2, "x": None}, ("ds", "b"): {"slice": "feat"}}
    assert slices == {(None, "7"): "f", ("ds", "b"): "row"}


def test_cli_and_web_report_a_malformed_feature_row(ws, tmp_path: Path, capsys) -> None:
    rows = _rows(ws["features"])
    rows[2]["dataset"] = []
    _write(ws["features"], rows)
    assert cli_main(["slice", "answerable", ws["records"], "--features", ws["features"]]) == 1
    assert _error_line(capsys).rstrip("\n").endswith("features.jsonl: row 3: 'dataset' must be a non-empty string or null, got list")
    r = _client(ws).get("/api/slices")
    assert r.status_code == 400
    assert r.json()["detail"].endswith("features.jsonl: row 3: 'dataset' must be a non-empty string or null, got list")


def test_cli_refuses_a_duplicated_feature_row(ws, capsys) -> None:
    rows = _rows(ws["features"])
    _write(ws["features"], [rows[0], *rows])
    assert cli_main(["slice", "mine", ws["records"], "--features", ws["features"]]) == 1
    assert "row 2 is the same item as row 1" in _error_line(capsys)


def _probe_rows(ws: dict[str, str]) -> list[dict]:
    return _rows(ws["probe_records"])


def _first(rows: list[dict], role: str) -> int:
    return next(i for i, r in enumerate(rows) if r.get("probe_role") == role)


def _probe_error(ws: dict[str, str], tmp_path: Path, rows: list[dict]) -> str:
    path = _write(tmp_path / "probe.jsonl", rows)
    with pytest.raises(ProbeInputError) as e:
        audit_probe_records(path)
    return str(e.value)


def test_probe_role_answered_with_another_primitive(ws, tmp_path: Path) -> None:
    rows = _probe_rows(ws)
    noul, twin = _first(rows, "noul"), _first(rows, "twin")
    rows[noul] = {**rows[twin], "probe_role": "noul", "item_id": rows[noul]["item_id"]}
    assert _probe_error(ws, tmp_path, rows) == f"line {noul + 1}: probe_role 'noul' needs a noul answer, got choice"


def test_twin_probe_without_a_yes_option(ws, tmp_path: Path) -> None:
    rows = _probe_rows(ws)
    twin = _first(rows, "twin")
    probs = rows[twin]["probabilities"]
    rows[twin]["probabilities"] = {"si": probs["yes"], "no": probs["no"]}
    rows[twin]["choice"] = "si" if rows[twin]["choice"] == "yes" else "no"
    if rows[twin].get("label") == "yes":
        rows[twin]["label"] = "si"
    assert _probe_error(ws, tmp_path, rows) == (
        f"line {twin + 1}: probe_role 'twin' needs a choice answer with a 'yes' option, got choice"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"probe_role": "sibling"}, "'probe_role' must be one of noul, twin, inverse"),
        ({"probe_role": None}, "'probe_role' must be one of noul, twin, inverse"),
        ({"probe_question": 1}, "'probe_question' must be a non-empty string, got int"),
        ({"probe_question": ""}, "'probe_question' must be a non-empty string, got str"),
        ({"probe_base_question_id": ["q"]}, "'probe_base_question_id' must be a non-empty string, got list"),
    ],
)
def test_probe_tag_schema(ws, tmp_path: Path, change: dict, message: str) -> None:
    rows = _probe_rows(ws)
    rows[0].update(change)
    assert _probe_error(ws, tmp_path, rows) == f"line 1: {message}"


def test_one_role_twice_for_an_item(ws, tmp_path: Path) -> None:
    rows = _probe_rows(ws)
    inverse = _first(rows, "inverse")
    rows.append(dict(rows[inverse]))
    assert _probe_error(ws, tmp_path, rows) == (
        f"line {len(rows)}: a second 'inverse' probe for the same item and model as line {inverse + 1}"
    )


def test_failed_and_untagged_records_are_skipped(ws, tmp_path: Path) -> None:
    rows = _probe_rows(ws)
    before = audit_probe_records(ws["probe_records"])
    # A failed probe is written with its tags and without an answer (monitor/runner.py).
    for role in ("twin", "noul"):
        failed = {k: v for k, v in rows[_first(rows, role)].items() if k not in ("p", "probabilities", "choice", "confidence")}
        rows.append({**failed, "model_served": None, "error_code": "timeout"})
    untagged = {k: v for k, v in rows[_first(rows, "noul")].items() if k not in ("probe_role", "probe_question")}
    rows.append(untagged)
    assert audit_probe_records(_write(tmp_path / "probe.jsonl", rows)) == before


def test_items_with_and_without_a_dataset_sort(ws, tmp_path: Path) -> None:
    rows = [r for r in _probe_rows(ws) if r["probe_role"] == "noul"][:2]
    rows[0].pop("dataset", None)
    rows[1]["dataset"] = "default"
    (report,) = audit_probe_records(_write(tmp_path / "probe.jsonl", rows))
    assert [(p["dataset"], p["item_id"]) for p in report["points"]] == [
        (None, rows[0]["item_id"]),
        ("default", rows[1]["item_id"]),
    ]


def test_cli_and_web_report_a_malformed_probe_row(ws, capsys) -> None:
    rows = _probe_rows(ws)
    rows[4]["probe_role"] = "sibling"
    _write(ws["probe_records"], rows)
    assert cli_main(["lint", "probe-audit", ws["probe_records"]]) == 1
    assert _error_line(capsys).rstrip("\n").endswith("probe_records.jsonl: line 5: 'probe_role' must be one of noul, twin, inverse")
    r = _client(ws).get("/api/lint")
    assert r.status_code == 400
    assert r.json()["detail"] == "probe_records.jsonl: line 5: 'probe_role' must be one of noul, twin, inverse"


def test_inverse_probe_answered_with_another_primitive(ws, tmp_path: Path) -> None:
    rows = _probe_rows(ws)
    inverse, twin = _first(rows, "inverse"), _first(rows, "twin")
    rows[inverse] = {**rows[twin], "probe_role": "inverse", "item_id": rows[inverse]["item_id"]}
    assert _probe_error(ws, tmp_path, rows) == f"line {inverse + 1}: probe_role 'inverse' needs a noul answer, got choice"


def test_line_numbers_count_blank_lines(ws, tmp_path: Path) -> None:
    rows = _probe_rows(ws)
    rows[0]["probe_role"] = "sibling"
    path = tmp_path / "probe.jsonl"
    path.write_text("\n  \n" + "".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ProbeInputError) as e:
        audit_probe_records(path)
    assert str(e.value).startswith("line 3: ")


def test_cli_and_web_skip_failed_probes(ws, capsys) -> None:
    rows = _probe_rows(ws)
    assert cli_main(["lint", "probe-audit", ws["probe_records"]]) == 0
    before = json.loads(capsys.readouterr().out)
    for role in ("noul", "twin", "inverse"):
        failed = {k: v for k, v in rows[_first(rows, role)].items() if k not in ("p", "probabilities", "choice", "confidence")}
        rows.append({**failed, "model_served": None, "error_code": "timeout"})
    _write(ws["probe_records"], rows)
    assert cli_main(["lint", "probe-audit", ws["probe_records"]]) == 0
    assert json.loads(capsys.readouterr().out) == before
    r = _client(ws).get("/api/lint")
    assert r.status_code == 200
    assert r.json()["probe_audit"] == before


@pytest.mark.parametrize(
    ("role", "change", "message"),
    [
        ("noul", {"probe_question": 1}, "'probe_question' must be a non-empty string, got int"),
        ("twin", {"probe_role": "sibling"}, "'probe_role' must be one of noul, twin, inverse"),
    ],
)
def test_cli_and_web_report_each_probe_error(ws, capsys, role: str, change: dict, message: str) -> None:
    rows = _probe_rows(ws)
    i = _first(rows, role)
    rows[i].update(change)
    _write(ws["probe_records"], rows)
    assert cli_main(["lint", "probe-audit", ws["probe_records"]]) == 1
    assert _error_line(capsys).rstrip("\n").endswith(f"probe_records.jsonl: line {i + 1}: {message}")
    r = _client(ws).get("/api/lint")
    assert r.status_code == 400
    assert r.json()["detail"] == f"probe_records.jsonl: line {i + 1}: {message}"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"item_id": True}, "row 3: 'item_id' must be a non-empty string or an integer, got bool"),
        ({"features": {"tags": ["a"]}}, "row 3: each 'features' value must be a string, number, boolean or null, got list"),
        ({"slice": 3}, "row 3: 'slice' must be a string, got int"),
    ],
)
def test_cli_and_web_report_each_feature_error(ws, capsys, change: dict, message: str) -> None:
    rows = _rows(ws["features"])
    rows[2].update(change)
    _write(ws["features"], rows)
    for sub in ("answerable", "mine"):
        assert cli_main(["slice", sub, ws["records"], "--features", ws["features"]]) == 1
        assert _error_line(capsys).rstrip("\n").endswith(f"features.jsonl: {message}")
    r = _client(ws).get("/api/slices")
    assert r.status_code == 400
    assert r.json()["detail"].endswith(f"features.jsonl: {message}")


def _old_feature_index(rows: list[dict]) -> tuple[dict, dict]:
    # The loop `s1 slice` and /api/slices ran before parse_feature_rows, kept as the baseline.
    features_by_item: dict = {}
    slice_by_item: dict = {}
    for row in rows:
        key = (row.get("dataset"), str(row.get("item_id", "")))
        feats = row.get("features", {})
        if isinstance(feats, dict):
            features_by_item[key] = feats
            if "slice" in feats:
                slice_by_item[key] = str(feats["slice"])
        if "slice" in row:
            slice_by_item[key] = str(row["slice"])
    return features_by_item, slice_by_item


def test_valid_feature_rows_index_as_before(ws) -> None:
    rows = _rows(ws["features"])
    rows.append({"item_id": 9001, "features": {"slice": "f", "n": 2.5, "flag": False, "x": None}})
    rows.append({"dataset": "other", "item_id": "z", "slice": "row-wins", "features": {"slice": "feat"}})
    assert parse_feature_rows(rows) == _old_feature_index(rows)


def _role_answered_as_choice(rows: list[dict], role: str) -> tuple[int, str]:
    i, twin = _first(rows, role), _first(rows, "twin")
    rows[i] = {**rows[twin], "probe_role": role, "item_id": rows[i]["item_id"]}
    return i, f"probe_role '{role}' needs a noul answer, got choice"


def _twin_without_yes(rows: list[dict]) -> tuple[int, str]:
    i = _first(rows, "twin")
    probs = rows[i]["probabilities"]
    rows[i] = {**rows[i], "probabilities": {"si": probs["yes"], "no": probs["no"]}, "choice": "no", "label": "no"}
    return i, "probe_role 'twin' needs a choice answer with a 'yes' option, got choice"


def _repeated_role(rows: list[dict]) -> tuple[int, str]:
    first = _first(rows, "inverse")
    rows.append(dict(rows[first]))
    return len(rows) - 1, f"a second 'inverse' probe for the same item and model as line {first + 1}"


def _bad_base_question(rows: list[dict]) -> tuple[int, str]:
    rows[0]["probe_base_question_id"] = ["q"]
    return 0, "'probe_base_question_id' must be a non-empty string, got list"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda rows: _role_answered_as_choice(rows, "noul"),
        lambda rows: _role_answered_as_choice(rows, "inverse"),
        _twin_without_yes,
        _repeated_role,
        _bad_base_question,
    ],
    ids=["noul-as-choice", "inverse-as-choice", "twin-without-yes", "repeated-role", "base-question-type"],
)
def test_cli_and_web_report_each_probe_schema_error(ws, capsys, corrupt) -> None:
    rows = _probe_rows(ws)
    i, message = corrupt(rows)
    _write(ws["probe_records"], rows)
    expected = f"probe_records.jsonl: line {i + 1}: {message}"
    assert cli_main(["lint", "probe-audit", ws["probe_records"]]) == 1
    assert _error_line(capsys).rstrip("\n").endswith(expected)
    r = _client(ws).get("/api/lint")
    assert r.status_code == 400
    assert r.json()["detail"] == expected


def test_web_refuses_a_duplicated_feature_row(ws) -> None:
    rows = _rows(ws["features"])
    _write(ws["features"], [rows[0], *rows])
    r = _client(ws).get("/api/slices")
    assert r.status_code == 400
    assert r.json()["detail"].endswith("features.jsonl: row 2 is the same item as row 1")
