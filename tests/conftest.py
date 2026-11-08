"""Suite-wide isolation from the developer's environment."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_inherited_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """`S1_DEBUG` makes `main()` re-raise; a test that wants it sets it itself."""
    monkeypatch.delenv("S1_DEBUG", raising=False)
