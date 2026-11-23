from mono_agent.config import Settings


def test_synthetic_mode_by_default(monkeypatch):
    monkeypatch.delenv("MONO_TOKEN", raising=False)
    assert not Settings(_env_file=None).use_real_api  # type: ignore[call-arg]


def test_token_never_in_repr(monkeypatch):
    monkeypatch.setenv("MONO_TOKEN", "super-secret")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.use_real_api
    assert "super-secret" not in repr(s)


def test_empty_token_means_synthetic_mode(monkeypatch):
    monkeypatch.setenv("MONO_TOKEN", "")
    assert not Settings(_env_file=None).use_real_api  # type: ignore[call-arg]
