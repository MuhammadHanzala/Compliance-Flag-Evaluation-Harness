from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep LangGraph from sending traces even if the shell sets LANGSMITH_TRACING."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
