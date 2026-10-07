"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _no_semantic_cache(monkeypatch, request):
    """Tests use fake retrievers/LLMs; keep the real semantic cache out unless a test opts in."""
    if request.node.get_closest_marker("semantic_cache"):
        return
    import agents.kb_agent as kb
    monkeypatch.setattr(kb, "_get_cache", lambda: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "semantic_cache: test uses the semantic cache")
