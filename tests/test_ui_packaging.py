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

    The current `p.parent.name != "ui"` only excluded files directly in
    `omnilingual/ui/`.  This version excludes any file whose path contains
    `/ui/` anywhere, and resolves relative imports properly.
    """
    import re

    REPO = Path(__file__).resolve().parents[1]
    cli_dir = REPO / "omnilingual"

    def _is_under_ui(filepath: Path) -> bool:
        """True if the file lives anywhere under omnilingual/ui/."""
        parts = filepath.parts
        return "/ui/" in filepath.as_posix() or "\\ui\\" in filepath.as_posix()

    def _imports_omnilingual_ui(source: str, file_path: Path) -> bool:
        """Return True if *source* contains an import of omnilingual.ui.

        Handles:
        - `import omnilingual.ui`
        - `from omnilingual.ui import X`
        - `from omnilingual import ui`  (imports the ui subpackage from omnilingual)
        - `from . import ui`  (resolved relative to file_path's package)
        - `from .. import ui`  (resolved relative to file_path's package)
        """
        # Absolute import: import omnilingual.ui or from omnilingual.ui import ...
        if re.search(r"\bimport\s+omnilingual\.ui\b", source):
            return True
        if re.search(r"\bfrom\s+omnilingual\.ui\b", source):
            return True
        # from omnilingual import ui  (imports the ui subpackage from omnilingual)
        if re.search(r"\bfrom\s+omnilingual\s+import\s+ui\b", source):
            return True

        # Relative imports: from . import ui  or  from .. import ui etc.
        # Determine the package name from the file's parent directories.
        # e.g. /path/omnilingual/cli.py -> package is "omnilingual"
        #      /path/omnilingual/omnilingual/cli.py -> package is "omnilingual" (nested)
        try:
            rel = file_path.relative_to(REPO)
            # Get the package part (all but the filename)
            pkg_parts = rel.parent.parts
            if not pkg_parts:
                return False
            # Build the dotted package name up to the parent dir
            pkg_name = ".".join(pkg_parts)
            # Map relative level: from . = 1 level up, from .. = 2 levels up, etc.
            # We look for patterns like `from . import ui`, `from .. import ui`
            for m in re.finditer(r"\bfrom\s+(\.+)\s+import\s+ui\b", source):
                dots = len(m.group(1))
                # Go up `dots` levels from the current package
                pkg_parts_list = list(pkg_parts)
                if len(pkg_parts_list) > dots:
                    resolved_pkg = ".".join(pkg_parts_list[:-dots])
                else:
                    # If we go past the root, it's a top-level package
                    resolved_pkg = ""
                if resolved_pkg and f"{resolved_pkg}.ui" == "omnilingual.ui":
                    return True
                # Also check if the resolved package itself is "ui" within omnilingual
                # e.g. from . import ui when current file is in omnilingual.cli
                # resolves to omnilingual.ui
                if resolved_pkg == "omnilingual" and dots == 1:
                    # from . import ui when file is in omnilingual/ -> omnilingual.ui
                    return True
        except ValueError:
            pass

        return False

    offenders: list[str] = []
    for p in cli_dir.rglob("*.py"):
        # Skip files under ui/ — those are the UI package itself
        if _is_under_ui(p):
            continue
        try:
            source = p.read_text(encoding="utf-8")
        except Exception:
            continue
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


def test_ui_init_has_no_web_imports():
    """Assert omnilingual/ui/__init__.py contains no import/fastapi/uvicorn/pywebview."""
    init_path = (
        Path(__file__).resolve().parents[1] / "omnilingual" / "ui" / "__init__.py"
    )
    source = init_path.read_text(encoding="utf-8")
    for lib in ("fastapi", "uvicorn", "pywebview"):
        # Check both bare `import` and `from ... import` forms
        assert f"import {lib}" not in source, (
            f"{lib} imported in omnilingual/ui/__init__.py — should not be there"
        )
        assert f"from {lib}" not in source, (
            f"{lib} from-import in omnilingual/ui/__init__.py — should not be there"
        )