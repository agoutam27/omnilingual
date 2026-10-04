import stat

import pytest

from omnilingual.ui import settings


def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return settings


def test_defaults_mirror_the_cli_flag_surface(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    assert settings.load() == settings.DEFAULTS


def test_defaults_are_exactly_the_specified_parameter_table():
    """Pins every value, so a mutated default fails here rather than in a run.

    Key-presence alone proved nothing: max_chunk_s could sit at 30.0 (above
    MAX_CHUNK_LIMIT_S, which the CLI rejects), stt at a backend that does not
    exist, and every test still passed.
    """
    assert settings.DEFAULTS == {
        "mode": "live",
        "source": "",
        "out": "standup.md",
        "stt": "sarvam",
        "stt_model": "",
        "mt": "mayura",
        "mt_model": "",
        "diarize": False,
        "num_speakers": 3,
        "langs": [],
        "english_only": False,
        "work_dir": "",
        "device": "Omnilingual",
        "mic_only": False,
        "target_s": 8.0,
        "max_chunk_s": 28.0,
        "min_chunk_s": 5.0,
        "noise_db": -35.0,
        "stt_workers": 2,
        "max_cost": 50.0,
    }


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


def test_load_falls_back_to_defaults_on_a_non_utf8_file(tmp_path, monkeypatch):
    """A byte-invalid store must not stop the panel opening.

    UnicodeDecodeError is a ValueError, not an OSError, so the OSError-only
    guard let a mangled ui.toml raise straight through a load() that promises
    never to raise.
    """
    _isolated(tmp_path, monkeypatch)
    path = settings.config_dir() / "ui.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'out = "\xff\xfe-cafe"\n')
    assert settings.load() == settings.DEFAULTS


def test_save_round_trips_booleans_as_booleans(tmp_path, monkeypatch):
    """Identity, not equality: a bool must not come back as the string "false".

    `diarize = "false"` is valid TOML and bool("false") is True, so a lost bool
    branch would silently switch speaker labelling — which costs money — back on.
    """
    _isolated(tmp_path, monkeypatch)
    settings.save({"diarize": True, "mic_only": False, "english_only": True})
    loaded = settings.load()
    assert loaded["diarize"] is True
    assert loaded["mic_only"] is False
    assert loaded["english_only"] is True


def test_save_round_trips_a_non_empty_lang_list(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.save({"langs": ["hi-IN", "ta-IN", "en-IN"]})
    assert settings.load()["langs"] == ["hi-IN", "ta-IN", "en-IN"]


def test_load_does_not_create_a_file(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.load()
    assert not settings.settings_path().exists()
