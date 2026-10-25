"""Shared helpers for turning pydantic validation failures into one-line messages."""

from pydantic import ValidationError


def pydantic_errors(error: ValidationError) -> list[str]:
    out = []
    for e in error.errors(include_url=False):
        loc = ".".join(str(part) for part in e["loc"]) or "(root)"
        out.append(f"{loc}: {e['msg']}")
    return out
