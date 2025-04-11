"""
Base classes for file management tools.
"""

from typing import Optional
from pydantic import BaseModel

from .file_paths import (
    FilePathResolver,
    OPERATION_READ,
    OPERATION_WRITE,
    FileNotFoundError,
    AccessDeniedError,
    FILE_NOT_FOUND_MESSAGE,
    ACCESS_DENIED_MESSAGE,
)


class BaseFileTool(BaseModel):
    """
    Base class for all file operation tools.
    Provides common path resolution functionality.
    """

    root_dir: Optional[str] = None
    """The final path will be chosen relative to root_dir if specified."""

    def _get_path_resolver(self) -> FilePathResolver:
        """Get a file path resolver with the current root directory."""
        return FilePathResolver(root_dir=self.root_dir)


class ReadFileTool(BaseFileTool):
    """
    Base class for read-only file operation tools.
    Allows reading from any readable location.
    """

    def resolve_path(self, file_path: str) -> str:
        """
        Resolve a path for read operations.
        Returns a string representation of the resolved path.
        """
        resolver = self._get_path_resolver()
        try:
            resolved_path = resolver.resolve_path(file_path, operation=OPERATION_READ)
            return str(resolved_path)
        except FileNotFoundError:
            return FILE_NOT_FOUND_MESSAGE.format(path=file_path)
        except Exception as e:
            return f"Error: {str(e)}"


class WriteFileTool(BaseFileTool):
    """
    Base class for write file operation tools.
    Restricts writing to workspace or whitelisted locations.
    """

    def resolve_path(self, file_path: str) -> str:
        """
        Resolve a path for write operations.
        Returns a string representation of the resolved path.
        """
        resolver = self._get_path_resolver()
        try:
            resolved_path = resolver.resolve_path(file_path, operation=OPERATION_WRITE)
            return str(resolved_path)
        except FileNotFoundError:
            return FILE_NOT_FOUND_MESSAGE.format(path=file_path)
        except AccessDeniedError:
            root_dir = resolver.get_effective_root()
            return ACCESS_DENIED_MESSAGE.format(path=file_path, allowed_dir=root_dir)
        except Exception as e:
            return f"Error: {str(e)}"
