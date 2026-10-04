import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_ui_extra_declares_the_three_web_dependencies():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "ui" in extras, "the ui extra must exist so the CLI gains no web dep"
    joined = " ".join(extras["ui"])
    assert "fastapi" in joined
    assert "uvicorn" in joined
    assert "pywebview" in joined


def test_console_script_points_at_the_ui_entry_point():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["omnilingual-ui"] == "omnilingual.ui.__main__:main"


def test_ui_package_is_not_imported_by_the_cli():
    # The CLI must never acquire a web dependency, so nothing outside ui/ may
    # reach omnilingual.ui on its import path.
    offenders = [p.name for p in (REPO / "omnilingual").rglob("*.py")
                 if p.parent.name != "ui"
                 and "omnilingual.ui" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_ui_package_imports_without_web_dependencies():
    import omnilingual.ui  # noqa: F401