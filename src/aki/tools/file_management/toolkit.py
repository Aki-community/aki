"""Toolkit for file management operations."""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils.pydantic import get_fields
from pydantic import model_validator

from .copy import CopyFileTool
from .delete import DeleteFileTool
from .file_search import FileSearchTool
from .list_dir import ListDirectoryTool
from .move import MoveFileTool
from .read import ReadFileTool
from .write import WriteFileTool
from .fast_edit import FastEditTool
from .set_workspace import SetWorkspaceTool

# Conditionally import GrepTool based on ripgrep availability
GrepTool = None


def is_ripgrep_available():
    """Check if ripgrep (rg) is available on the system."""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, text=True, check=False)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


if is_ripgrep_available():
    from .grep_tool import GrepTool

# All file tools (without grep initially)
_FILE_TOOLS: List[Type[BaseTool]] = [
    CopyFileTool,
    DeleteFileTool,
    FileSearchTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    FastEditTool,
    SetWorkspaceTool,
]

# Read-only tools that don't modify the filesystem (without grep initially)
_FILE_READ_ONLY_TOOLS: List[Type[BaseTool]] = [
    FileSearchTool,
    ReadFileTool,
    ListDirectoryTool,
    SetWorkspaceTool,
]

# Add GrepTool if available
if GrepTool is not None:
    _FILE_TOOLS.append(GrepTool)
    _FILE_READ_ONLY_TOOLS.append(GrepTool)

# Create tool maps
_FILE_TOOLS_MAP: Dict[str, Type[BaseTool]] = {
    get_fields(tool_cls)["name"].default: tool_cls for tool_cls in _FILE_TOOLS
}

_FILE_TOOLS_READ_ONLY_MAP: Dict[str, Type[BaseTool]] = {
    get_fields(tool_cls)["name"].default: tool_cls for tool_cls in _FILE_READ_ONLY_TOOLS
}


class FileManagementToolkit(BaseToolkit):
    """Toolkit for interacting with local files.

    *Security Notice*: This toolkit provides methods to interact with local files.
        If providing this toolkit to an agent on an LLM, ensure you scope
        the agent's permissions to only include the necessary permissions
        to perform the desired operations.

        By **default** the agent will have access to all files within
        the root dir and will be able to Copy, Delete, Move, Read, Write
        and List files in that directory.

        Consider the following:
        - Limit access to particular directories using `root_dir`.
        - Use filesystem permissions to restrict access and permissions to only
          the files and directories required by the agent.
        - Limit the tools available to the agent to only the file operations
          necessary for the agent's intended use.
        - Sandbox the agent by running it in a container.

    Parameters:
        root_dir: Optional. The root directory to perform file operations.
            If not provided, file operations are performed relative to the current
            working directory.
        selected_tools: Optional. The tools to include in the toolkit. If not
            provided, all tools are included.
    """

    root_dir: Optional[str] = None
    """If specified, all file operations are made relative to root_dir."""
    selected_tools: Optional[List[str]] = None
    """If provided, only provide the selected tools. Defaults to all."""

    @model_validator(mode="before")
    @classmethod
    def validate_tools(cls, values: dict) -> Any:
        selected_tools = values.get("selected_tools") or []
        for tool_name in selected_tools:
            if tool_name not in _FILE_TOOLS_MAP:
                raise ValueError(
                    f"File Tool of name {tool_name} not supported."
                    f" Permitted tools: {list(_FILE_TOOLS_MAP)}"
                )
        return values

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        allowed_tools = self.selected_tools or list(_FILE_TOOLS_MAP.keys())
        tools: List[BaseTool] = []
        for tool_name in allowed_tools:
            if tool_name in _FILE_TOOLS_MAP:
                tool_cls = _FILE_TOOLS_MAP[tool_name]
                tools.append(tool_cls(root_dir=self.root_dir))  # type: ignore[call-arg]
        return tools

    def get_read_only_tools(self) -> List[BaseTool]:
        """Get only the read-only tools in the toolkit."""
        allowed_tools = [
            name
            for name in _FILE_TOOLS_READ_ONLY_MAP.keys()
            if self.selected_tools is None or name in self.selected_tools
        ]
        tools: List[BaseTool] = []
        for tool_name in allowed_tools:
            if tool_name in _FILE_TOOLS_MAP:
                tool_cls = _FILE_TOOLS_MAP[tool_name]
                tools.append(tool_cls(root_dir=self.root_dir))  # type: ignore[call-arg]
        return tools


__all__ = ["FileManagementToolkit"]
