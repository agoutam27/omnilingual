"""Local desktop-style UI over the omnilingual pipeline.

Imported by nothing in the CLI: this package, and the FastAPI/pywebview
dependencies behind the `ui` extra, must never sit on the CLI's import path or
`uv run omnilingual transcribe` would acquire a web dependency.
"""
