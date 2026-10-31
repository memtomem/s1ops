"""Shared helpers for turning malformed input into one-line messages that name their file."""

import json
from pathlib import Path

from pydantic import ValidationError


def pydantic_errors(error: ValidationError) -> list[str]:
    out = []
    for e in error.errors(include_url=False):
        loc = ".".join(str(part) for part in e["loc"]) or "(root)"
        out.append(f"{loc}: {e['msg']}")
    return out


class _NonFinite(Exception):
    pass


def _non_finite(name: str) -> float:
    raise _NonFinite(name)


def read_json_file(path: str | Path, error: type[ValueError] = ValueError) -> object:
    """The JSON document in `path`. A malformed file raises `error` with a message that starts
    with the path: invalid UTF-8, invalid JSON, or a non-finite constant (`NaN`, `Infinity`),
    which Python's `json` accepts but JSON does not, an integer past Python's digit limit, or
    nesting too deep for the decoder. An unreadable file raises its `OSError`."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f, parse_constant=_non_finite)
    except UnicodeDecodeError as e:
        raise error(f"{path}: not valid UTF-8: {e}") from None
    except _NonFinite as e:
        raise error(f"{path}: non-finite number {e} is not valid JSON") from None
    except ValueError as e:  # JSONDecodeError, or the decoder's integer-digit limit
        raise error(f"{path}: not valid JSON: {e}") from None
    except RecursionError:
        raise error(f"{path}: not valid JSON: nested too deeply") from None
