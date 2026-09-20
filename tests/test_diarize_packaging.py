import tomllib
from pathlib import Path


def _pyproject() -> dict:
    root = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with root.open("rb") as f:
        return tomllib.load(f)


def test_diarize_extra_declared():
    extras = _pyproject()["project"]["optional-dependencies"]
    joined = " ".join(extras["diarize"])
    assert "sherpa-onnx" in joined
    assert "soundfile" in joined


def test_diarize_marker_declared():
    markers = _pyproject()["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("diarize:") for m in markers)
