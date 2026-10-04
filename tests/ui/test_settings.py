import stat

import pytest

from omnilingual.ui import settings


def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return settings


def test_defaults_mirror_the_cli_flag_surface(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    assert settings.load() == settings.DEFAULTS


def test_defaults_cover_every_panel_control():
    for key in ("mode", "source", "out", "stt", "stt_model", "mt", "mt_model",
                "diarize", "num_speakers", "langs", "english_only", "work_dir",
                "device", "mic_only", "target_s", "max_chunk_s", "min_chunk_s",
                "noise_db", "stt_workers", "max_cost"):
        assert key in settings.DEFAULTS, f"{key} missing from DEFAULTS"


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.save({"out": "/tmp/notes.md", "stt": "groq", "num_speakers": 4})
    loaded = settings.load()
    assert loaded["out"] == "/tmp/notes.md"
    assert loaded["stt"] == "groq"
    assert loaded["num_speakers"] == 4
    # Untouched keys keep their defaults rather than becoming None.
    assert loaded["device"] == "Omnilingual"


def test_save_rejects_unknown_keys(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as exc:
        settings.save({"nope": 1})
    assert "nope" in str(exc.value)


def test_save_never_writes_an_unknown_key(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        settings.save({"out": "a.md", "nope": 1})
    # The rejection happens before any write, so the file may not exist at all;
    # what must never happen is the rejected key reaching disk.
    path = settings.settings_path()
    assert not path.exists() or "nope" not in path.read_text(encoding="utf-8")


def test_settings_file_is_private(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.save({"out": "a.md"})
    path = settings.settings_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(settings.config_dir().stat().st_mode) == 0o700


def test_save_round_trips_a_value_holding_both_quote_characters(tmp_path, monkeypatch):
    """Paths are not Python identifiers and may hold both quote characters.

    Emitting such a value as a TOML literal string (``'it\\'s "q".md'``) produces
    a file tomllib rejects, because a literal string gives ``\\`` no meaning.
    load() swallows that rejection and hands back defaults, so a save that
    reported success would silently lose every setting on the next open.
    """
    _isolated(tmp_path, monkeypatch)
    settings.save({"out": "it's \"final\".md", "langs": ["it's", 'say "hi"']})
    loaded = settings.load()
    assert loaded["out"] == "it's \"final\".md"
    assert loaded["langs"] == ["it's", 'say "hi"']


def test_load_falls_back_to_defaults_on_corrupt_file(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.save({"out": "a.md"})
    settings.settings_path().write_text("this is not = valid toml [[[",
                                        encoding="utf-8")
    assert settings.load() == settings.DEFAULTS


def test_load_does_not_create_a_file(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.load()
    assert not settings.settings_path().exists()
