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
    """Assert no file outside omnilingual/ui imports from omnilingual.ui.

    Guards against both absolute and relative imports that would couple the
    CLI to the web UI package.  Uses regex-matched import resolution so that
    `from . import ui` is correctly resolved to `omnilingual.ui` relative to
    the importing file's package, and `from omnilingual.ui import X` is caught.

    The guard excludes any file whose path (relative to the package root)
    contains `/ui/`, and resolves relative imports by ascending the correct
    number of package levels so that `from . import ui` from `cli.py` maps to
    `omnilingual.ui` while `from . import ui` from `audio/` maps to
    `omnilingual.audio.ui` (not the target).
    """
    import re

    REPO = Path(__file__).resolve().parents[1]
    cli_dir = REPO / "omnilingual"

    def _is_under_ui(filepath: Path) -> bool:
        """True if the file lives anywhere under omnilingual/ui/ (relative to repo root)."""
        try:
            rel = filepath.relative_to(REPO)
        except ValueError:
            return False
        return "/ui/" in rel.as_posix() or "\\ui\\" in rel.as_posix()

    def _resolve_pkg_relative(pkg_parts, dots):
        """Resolve the package after ascending `dots` levels.

        e.g. from . import ui from cli.py (pkg_parts=("omnilingual",), dots=1)
        -> "omnilingual"; from . import ui from audio/foo.py -> "omnilingual";
        from .. import ui from audio/live.py -> "omnilingual".
        """
        if dots <= 0 or dots > len(pkg_parts):
            return ""
        resolved = pkg_parts[: len(pkg_parts) - dots + 1]
        return ".".join(resolved)

    def _imports_omnilingual_ui(source, file_path):
        """Return True if *source* contains an import of omnilingual.ui."""
        try:
            rel = file_path.relative_to(REPO)
        except ValueError:
            return False

        pkg_parts = rel.parent.parts
        if not pkg_parts:
            return False

        # Absolute import: import omnilingual.ui or from omnilingual.ui import ...
        if re.search(r"\bimport\s+omnilingual\.ui\b", source):
            return True
        if re.search(r"\bfrom\s+omnilingual\.ui\b", source):
            return True
        # from omnilingual import ui  (imports the ui subpackage from omnilingual)
        if re.search(r"\bfrom\s+omnilingual\s+import\s+ui\b", source):
            return True

        # --- from . import ui  (single dot) ---
        for m in re.finditer(r"\bfrom\s+(\.+)\s+import\s+ui\b", source):
            dots = len(m.group(1))
            resolved_pkg = _resolve_pkg_relative(pkg_parts, dots)
            if resolved_pkg and f"{resolved_pkg}.ui" == "omnilingual.ui":
                return True
            if resolved_pkg == "omnilingual" and dots == 1:
                return True

        # --- from .ui import X  (ui as subpackage of current package) ---
        if re.search(r"\bfrom\s+\.ui\s+import\b", source):
            resolved_pkg = _resolve_pkg_relative(pkg_parts, 1)
            if resolved_pkg and f"{resolved_pkg}.ui" == "omnilingual.ui":
                return True

        # --- from ..ui import X  (ui as subpackage of parent package) ---
        if re.search(r"\bfrom\s+\.\.ui\s+import\b", source):
            resolved_pkg = _resolve_pkg_relative(pkg_parts, 2)
            if resolved_pkg and f"{resolved_pkg}.ui" == "omnilingual.ui":
                return True

        return False

    offenders = []
    for p in cli_dir.rglob("*.py"):
        # Skip files under ui/ — those are the UI package itself
        if _is_under_ui(p):
            continue
        try:
            source = p.read_text(encoding="utf-8")
        except (PermissionError, OSError):
            offenders.append(f"{p.name} (unreadable)")
            continue
        except Exception:
            raise
        if _imports_omnilingual_ui(source, p):
            offenders.append(p.name)

    assert offenders == [], (
        f"Found {len(offenders)} file(s) outside omnilingual/ui importing "
        f"omnilingual.ui: {offenders}"
    )


def test_web_deps_absent_from_all_sections_except_ui_extra():
    """Assert fastapi/uvicorn/pywebview appear ONLY in optional-dependencies.ui.

    They must NOT be in project.dependencies, not in the dev group, and not
    in any other extra.  This keeps the phase's central invariant honest:
    the CLI never acquires a web dependency.
    """
    pj = _pyproject()

    # 1) project.dependencies must not contain any of the three
    deps = pj.get("project", {}).get("dependencies", [])
    dep_text = " ".join(deps)
    for lib in ("fastapi", "uvicorn", "pywebview"):
        assert lib not in dep_text, (
            f"{lib} found in [project].dependencies — should be ui-extra only"
        )

    # 2) dev dependency-group must not contain any of the three
    dev = pj.get("dependency-groups", {}).get("dev", [])
    dev_text = " ".join(dev)
    for lib in ("fastapi", "uvicorn", "pywebview"):
        assert lib not in dev_text, (
            f"{lib} found in [dependency-groups].dev — should be ui-extra only"
        )

    # 3) every other extra (non-ui) must not contain any of the three
    extras = pj["project"]["optional-dependencies"]
    for extra_name, extra_deps in extras.items():
        if extra_name == "ui":
            continue
        joined = " ".join(extra_deps)
        for lib in ("fastapi", "uvicorn", "pywebview"):
            assert lib not in joined, (
                f"{lib} found in extra '{extra_name}' — should be ui-extra only"
            )


def test_ui_package_is_a_real_module_not_namespace():
    """Assert omnilingual.ui is a real module with __file__, not a namespace dir."""
    import omnilingual.ui
    assert omnilingual.ui.__file__ is not None, (
        "omnilingual.ui must be a real package with __file__, not a namespace package"
    )


def test_ui_package_has_no_web_imports():
    """Assert no module in omnilingual/ui/ imports fastapi, uvicorn or pywebview.

    Scans the whole package rather than just the initializer: `ui.toml` settings
    and `.env` handling are stdlib-only precisely so they stay importable without
    the extra, and a web import in either would put the CLI back on that path.
    """
    ui_dir = Path(__file__).resolve().parents[1] / "omnilingual" / "ui"
    modules = sorted(ui_dir.glob("*.py"))
    assert modules, f"no modules found in {ui_dir}"
    for path in modules:
        source = path.read_text(encoding="utf-8")
        for lib in ("fastapi", "uvicorn", "pywebview"):
            # Check both bare `import` and `from ... import` forms
            assert f"import {lib}" not in source, (
                f"{lib} imported in omnilingual/ui/{path.name} — should not be there"
            )
            assert f"from {lib}" not in source, (
                f"{lib} from-import in omnilingual/ui/{path.name} — should not be there"
            )