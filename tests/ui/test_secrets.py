import stat

import pytest

from omnilingual.ui import secrets

FIXTURE = """# Omnilingual secrets — gitignored. Used via: uv run --env-file .env
# Groq: console.groq.com/keys
GROQ_API_KEY=groq-existing
# Gemini: aistudio.google.com/apikey
GEMINI_API_KEY=gemini-existing
"""


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(tmp_path / ".env"))
    path = secrets.env_path()
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def test_env_path_is_the_repo_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNILINGUAL_ENV_FILE", raising=False)
    path = secrets.env_path()
    assert path.name == ".env"
    assert path.parent.name == "omnilingual"


def test_present_reports_booleans_only(env_file):
    result = secrets.present()
    assert result == {"SARVAM_API_KEY": False, "GROQ_API_KEY": True,
                      "GEMINI_API_KEY": True}
    assert all(isinstance(v, bool) for v in result.values())


def test_set_key_appends_and_preserves_everything(env_file):
    secrets.set_key("SARVAM_API_KEY", "sarvam-new")
    text = env_file.read_text(encoding="utf-8")
    assert "sarvam-new" in text
    assert "GROQ_API_KEY=groq-existing" in text
    assert "GEMINI_API_KEY=gemini-existing" in text
    assert text.count("# Omnilingual secrets") == 1
    assert text.count("# Groq:") == 1


def test_set_key_replaces_in_place_without_duplicating(env_file):
    secrets.set_key("GROQ_API_KEY", "groq-replaced")
    text = env_file.read_text(encoding="utf-8")
    assert text.count("GROQ_API_KEY=") == 1
    assert "GROQ_API_KEY=groq-replaced" in text
    assert "groq-existing" not in text


def test_set_key_rejects_a_name_outside_the_known_set(env_file):
    with pytest.raises(ValueError):
        secrets.set_key("AWS_SECRET_ACCESS_KEY", "x")
    assert "AWS_SECRET" not in env_file.read_text(encoding="utf-8")


def test_set_key_keeps_mode_0600(env_file):
    env_file.chmod(0o600)
    secrets.set_key("SARVAM_API_KEY", "v")
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_clear_removes_the_line_and_keeps_others(env_file):
    secrets.clear("GROQ_API_KEY")
    text = env_file.read_text(encoding="utf-8")
    assert "GROQ_API_KEY" not in text
    assert "GEMINI_API_KEY=gemini-existing" in text
    assert text.count("# Gemini:") == 1


def test_clear_is_a_no_op_when_absent(env_file):
    before = env_file.read_text(encoding="utf-8")
    secrets.clear("SARVAM_API_KEY")
    assert env_file.read_text(encoding="utf-8") == before


def test_set_key_creates_the_file_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(tmp_path / "fresh" / ".env"))
    secrets.set_key("GEMINI_API_KEY", "g")
    path = secrets.env_path()
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "GEMINI_API_KEY=g" in path.read_text(encoding="utf-8")
