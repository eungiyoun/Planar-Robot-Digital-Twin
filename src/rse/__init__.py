"""Robotic Software Engineering Package."""

from importlib import metadata
from typing import Final

try:
    # Use the exact project name from your pyproject.toml!
    __version__: Final[str] = metadata.version("2026-spring-me498-group-2")
except metadata.PackageNotFoundError:
    # If we are just developing locally and it isn't "installed" yet, use a fallback
    __version__: Final[str] = "0.1.0"
