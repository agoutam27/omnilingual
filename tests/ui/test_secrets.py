import shutil
import stat
import subprocess

import pytest

from omnilingual.ui import secrets

FIXTURE = """# Omnilingual secrets — gitignored. Used via: uv run --env-file .env
# Groq: console.groq.com/keys
GROQ_API_KEY=groq-existing
# Gemini: aistudio.google.com/apikey
GEMINI_API_KEY=gemini-existing
"""


def _redirect(monkeypatch, directory, name=".env"):
    """Point env_path() at a throwaway file and prove the override took.

    Every test that writes a .env comes through here, not only the ones using the
    env_file fixture. An env_path() that ignores OMNILINGUAL_ENV_FILE — a
    plausible typo, and the shape of a mutation that let a whole suite pass while
    the fixture overwrote the developer's real .env — would otherwise aim these
    writes at the repo root and destroy two live keys.
    """
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(directory / name))
    path = secrets.env_path()
    assert path == directory / name, "OMNILINGUAL_ENV_FILE override not honoured"
    return path


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = _redirect(monkeypatch, tmp_path)
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def test_the_override_actually_redirects_env_path(tmp_path, monkeypatch):
    """The seam every other test in this file stands on.

    Asserted directly so that breaking env_path()'s override is a loud failure
    rather than a silently redirected write.
    """
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(tmp_path / ".env"))
    assert secrets.env_path() == tmp_path / ".env"


def test_leftover_temp_env_files_are_gitignored():
    """A kill between mkstemp and os.replace leaves the whole secret on disk.

    _write names its temporary .env-<random> in the .env's own directory, so
    .gitignore must cover that pattern: matching .env alone leaves the sibling
    untracked-but-stageable, and `git add -A` would put a live key in the index.
    """
    probe = secrets.REPO_ROOT / ".env-pytest-probe"
    assert not probe.exists(), f"{probe.name} must not exist for this check"
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe.name],
        cwd=secrets.REPO_ROOT, capture_output=True, check=False,
    )
    assert result.returncode == 0, (
        f"{probe.name} is not gitignored — a leftover temp .env holding a live "
        f"key could be staged by `git add -A`"
    )


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


def test_set_key_rewrites_every_duplicate_so_no_secret_survives(tmp_path, monkeypatch):
    """A hand-duplicated key must not leave a live secret behind a rotation.

    scripts/setup-mac.sh's env_set rewrites every matching line (its loop has
    no break). Rewriting only the first would leave the second copy holding the
    old value — the key the user believes they rotated away.
    """
    path = _redirect(monkeypatch, tmp_path)
    path.write_text(
        "# Omnilingual secrets\n"
        "GROQ_API_KEY=groq-old-one\n"
        "\n"
        "# a second, hand-added copy\n"
        "GROQ_API_KEY=groq-old-two\n"
        "GEMINI_API_KEY=gemini-existing\n",
        encoding="utf-8",
    )
    secrets.set_key("GROQ_API_KEY", "groq-new")
    text = path.read_text(encoding="utf-8")
    assert "groq-old-one" not in text
    assert "groq-old-two" not in text
    assert text.count("GROQ_API_KEY=groq-new") == 2
    assert text == (
        "# Omnilingual secrets\n"
        "GROQ_API_KEY=groq-new\n"
        "\n"
        "# a second, hand-added copy\n"
        "GROQ_API_KEY=groq-new\n"
        "GEMINI_API_KEY=gemini-existing\n"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_clear_removes_every_duplicate_line(tmp_path, monkeypatch):
    path = _redirect(monkeypatch, tmp_path)
    path.write_text(
        "# keep me\nGROQ_API_KEY=one\nGEMINI_API_KEY=gemini-existing\nGROQ_API_KEY=two\n",
        encoding="utf-8",
    )
    secrets.clear("GROQ_API_KEY")
    text = path.read_text(encoding="utf-8")
    assert "GROQ_API_KEY" not in text
    assert text == "# keep me\nGEMINI_API_KEY=gemini-existing\n"


def test_export_prefixed_lines_are_treated_as_the_same_key(tmp_path, monkeypatch):
    """`export NAME=` counts everywhere, or the three operations disagree.

    setup-mac.sh's pattern does not match these lines at all; treating them as
    real keys is the more useful behaviour, but it has to be uniform or a
    rotation can leave a stale copy that present() still counts as set.
    """
    path = _redirect(monkeypatch, tmp_path)
    path.write_text(
        "export GROQ_API_KEY='groq-exported'\nGROQ_API_KEY=groq-plain\n",
        encoding="utf-8",
    )
    assert secrets.present()["GROQ_API_KEY"] is True
    secrets.set_key("GROQ_API_KEY", "groq-new")
    text = path.read_text(encoding="utf-8")
    assert "groq-exported" not in text
    assert "groq-plain" not in text
    assert text.count("GROQ_API_KEY=groq-new") == 2
    secrets.clear("GROQ_API_KEY")
    assert "GROQ_API_KEY" not in path.read_text(encoding="utf-8")


def test_present_answers_for_the_last_duplicate_like_env_get(tmp_path, monkeypatch):
    """env_get takes `tail -n 1`, and so does a dotenv loader.

    A trailing blank assignment therefore clears the key for the CLI, so it must
    read as absent here too — otherwise the panel offers to fill in a key the
    run would not use.
    """
    path = _redirect(monkeypatch, tmp_path)
    path.write_text("GROQ_API_KEY=groq-one\nGROQ_API_KEY=\n", encoding="utf-8")
    assert secrets.present()["GROQ_API_KEY"] is False
    path.write_text("GROQ_API_KEY=\nGROQ_API_KEY=groq-two\n", encoding="utf-8")
    assert secrets.present()["GROQ_API_KEY"] is True


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
    path = _redirect(monkeypatch, tmp_path / "fresh")
    secrets.set_key("GEMINI_API_KEY", "g")
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "GEMINI_API_KEY=g" in path.read_text(encoding="utf-8")


def test_present_reports_all_false_for_a_non_utf8_file(tmp_path, monkeypatch):
    """A .env that is not UTF-8 reads as no keys, not as a crash.

    UnicodeDecodeError is a ValueError, not an OSError, so catching only OSError
    made a stray latin-1 byte in a comment take the whole panel down.
    """
    path = _redirect(monkeypatch, tmp_path)
    path.write_bytes(b"# caf\xe9\nGROQ_API_KEY=g\n")
    assert secrets.present() == dict.fromkeys(secrets.KEYS, False)


def test_set_key_trims_a_pasted_value(env_file):
    """A clipboard newline is the common case and must not leave a blank line."""
    secrets.set_key("GROQ_API_KEY", "groq-pasted\n")
    text = env_file.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=groq-pasted\n# Gemini:" in text
    assert "groq-pasted\n\n" not in text


def test_set_key_refuses_a_value_that_would_inject_a_second_key(env_file):
    """A newline plus content would silently add another key to the CLI's .env.

    The error may name the key but never the value, so a rejection cannot leak
    the secret into a log or a traceback.
    """
    before = env_file.read_bytes()
    with pytest.raises(ValueError) as exc:
        secrets.set_key("GROQ_API_KEY", "abc\nSARVAM_API_KEY=INJECTED")
    assert "GROQ_API_KEY" in str(exc.value)
    assert "INJECTED" not in str(exc.value)
    assert env_file.read_bytes() == before


def test_set_key_keeps_crlf_line_endings_on_the_lines_it_does_not_touch(tmp_path, monkeypatch):
    """env_set copies every other line verbatim, \r and all.

    str.splitlines() also breaks on \\r, \\x0b, \\x85 and \\u2028, so reading a
    CRLF .env that way and writing it back silently rewrote the whole file's line
    endings on the first set_key.
    """
    path = _redirect(monkeypatch, tmp_path)
    path.write_bytes(b"GROQ_API_KEY=groq-one\r\nOTHER=1\r\n")
    secrets.set_key("GROQ_API_KEY", "groq-new")
    assert path.read_bytes() == b"GROQ_API_KEY=groq-new\nOTHER=1\r\n"


def test_set_key_does_not_split_on_a_unicode_line_separator(tmp_path, monkeypatch):
    """\\u2028 is a line break to str.splitlines() but not to a .env parser.

    Rotating such a file injected an orphan line holding the tail of the old
    value into the user's .env.
    """
    path = _redirect(monkeypatch, tmp_path)
    path.write_text("GROQ_API_KEY=a\u2028b\nOTHER=1\n", encoding="utf-8")
    secrets.set_key("GROQ_API_KEY", "rotated")
    assert path.read_text(encoding="utf-8") == "GROQ_API_KEY=rotated\nOTHER=1\n"


def test_repeated_set_key_calls_do_not_accumulate_blank_lines(tmp_path, monkeypatch):
    path = _redirect(monkeypatch, tmp_path)
    path.write_text("GROQ_API_KEY=one\n", encoding="utf-8")
    for index in range(5):
        secrets.set_key("GROQ_API_KEY", f"v{index}")
    assert path.read_text(encoding="utf-8") == "GROQ_API_KEY=v4\n"
