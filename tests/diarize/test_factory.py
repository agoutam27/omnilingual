import pytest

from omnilingual.config import ConfigError, load_settings
from omnilingual.diarize import build_diarizer
from omnilingual.diarize.sherpa import SherpaDiarizer


def test_sherpa_selected_by_default_flag(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    dz = build_diarizer(load_settings(api_key="k", env={}, diarizer="sherpa"))
    assert isinstance(dz, SherpaDiarizer)
    assert dz.model.startswith("sherpa:")


def test_sherpa_without_extra_explains_install(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="diarize"):
        build_diarizer(load_settings(api_key="k", env={}, diarizer="sherpa"))
