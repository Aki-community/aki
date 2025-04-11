"""Whitelist configuration for file operations."""

from pathlib import Path
from typing import Set

# List of directories that are always allowed for file operations
WHITELISTED_PATHS: Set[Path] = {Path.home() / ".aki"}


def get_whitelisted_paths() -> Set[Path]:
    """Get the set of whitelisted paths."""
    return WHITELISTED_PATHS


def is_path_whitelisted(path: Path) -> bool:
    """Check if a path is whitelisted or within a whitelisted directory."""
    path = path.resolve()

    # Check if path is the same as any whitelisted path or within a whitelisted directory
    for whitelist_path in WHITELISTED_PATHS:
        whitelist_path = whitelist_path.resolve()
        try:
            if path == whitelist_path or path.is_relative_to(whitelist_path):
                return True
        except (ValueError, AttributeError):
            # For Python < 3.9 compatibility, handle is_relative_to manually
            try:
                path.relative_to(whitelist_path)
                return True
            except ValueError:
                pass

    return False
