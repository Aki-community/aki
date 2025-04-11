"""Tests for the file management tools."""

import os
import sys
import pytest
import tempfile
from pathlib import Path

from aki.tools.file_management.file_paths import (
    FilePathResolver,
    OPERATION_READ,
    OPERATION_WRITE,
    FileNotFoundError,
    AccessDeniedError,
)
from aki.tools.file_management.base_tools import (
    BaseFileTool,
    ReadFileTool,
    WriteFileTool,
)
from aki.tools.file_management.read import ReadFileTool as ReadFileImplementation
from aki.tools.file_management.write import (
    WriteFileTool as WriteFileImplementation,
)
from aki.tools.file_management.list_dir import ListDirectoryTool


@pytest.fixture
def test_environment():
    """Create test directories and files for permission testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure for testing
        workspace_dir = Path(tmpdir) / "workspace"
        outside_dir = Path(tmpdir) / "outside"

        # Create the directories
        workspace_dir.mkdir()
        outside_dir.mkdir()

        # Create test files
        workspace_file = workspace_dir / "test_file.txt"
        workspace_file.write_text("This is a test file in the workspace")

        outside_file = outside_dir / "outside_file.txt"
        outside_file.write_text("This is a test file outside the workspace")

        nonexistent_file = outside_dir / "nonexistent.txt"

        # Create resolver and tool instances
        resolver = FilePathResolver(root_dir=str(workspace_dir))
        read_tool = ReadFileImplementation(root_dir=str(workspace_dir))
        write_tool = WriteFileImplementation(root_dir=str(workspace_dir))
        list_tool = ListDirectoryTool(root_dir=str(workspace_dir))

        yield {
            "workspace_dir": workspace_dir,
            "outside_dir": outside_dir,
            "workspace_file": workspace_file,
            "outside_file": outside_file,
            "nonexistent_file": nonexistent_file,
            "resolver": resolver,
            "read_tool": read_tool,
            "write_tool": write_tool,
            "list_tool": list_tool,
        }


def test_path_resolver_read_workspace(test_environment):
    """Test path resolver with read operations in workspace."""
    resolver = test_environment["resolver"]
    workspace_file = test_environment["workspace_file"]

    resolved_path = resolver.resolve_path(str(workspace_file), operation=OPERATION_READ)
    assert resolved_path.samefile(workspace_file)


def test_path_resolver_read_outside(test_environment):
    """Test path resolver with read operations outside workspace."""
    resolver = test_environment["resolver"]
    outside_file = test_environment["outside_file"]

    resolved_path = resolver.resolve_path(str(outside_file), operation=OPERATION_READ)
    assert resolved_path.samefile(outside_file)


def test_path_resolver_read_nonexistent(test_environment):
    """Test path resolver with nonexistent files."""
    resolver = test_environment["resolver"]
    nonexistent_file = test_environment["nonexistent_file"]

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        resolver.resolve_path(str(nonexistent_file), operation=OPERATION_READ)


def test_path_resolver_write_workspace(test_environment):
    """Test path resolver with write operations in workspace."""
    resolver = test_environment["resolver"]
    workspace_dir = test_environment["workspace_dir"]
    new_file = workspace_dir / "new_file.txt"

    resolved_path = resolver.resolve_path(str(new_file), operation=OPERATION_WRITE)
    # Check that the paths refer to the same file, even if one has /private prefix on macOS
    assert resolved_path.name == new_file.name
    assert resolved_path.parent.name == new_file.parent.name


def test_path_resolver_write_outside(test_environment):
    """Test path resolver with write operations outside workspace."""
    resolver = test_environment["resolver"]
    outside_dir = test_environment["outside_dir"]
    outside_file = outside_dir / "new_outside_file.txt"

    # Should raise AccessDeniedError
    with pytest.raises(AccessDeniedError):
        resolver.resolve_path(str(outside_file), operation=OPERATION_WRITE)


def test_read_file_tool_workspace(test_environment):
    """Test ReadFileTool with files in workspace."""
    read_tool = test_environment["read_tool"]
    workspace_file = test_environment["workspace_file"]

    result = read_tool._run(str(workspace_file))
    assert "This is a test file in the workspace" in result


def test_read_file_tool_outside(test_environment):
    """Test ReadFileTool with files outside workspace."""
    read_tool = test_environment["read_tool"]
    outside_file = test_environment["outside_file"]

    result = read_tool._run(str(outside_file))
    assert "This is a test file outside the workspace" in result


def test_read_file_tool_nonexistent(test_environment):
    """Test ReadFileTool with nonexistent files."""
    read_tool = test_environment["read_tool"]
    nonexistent_file = test_environment["nonexistent_file"]

    result = read_tool._run(str(nonexistent_file))
    assert "No such file or directory" in result


def test_write_file_tool_workspace(test_environment):
    """Test WriteFileTool with files in workspace."""
    write_tool = test_environment["write_tool"]
    workspace_dir = test_environment["workspace_dir"]
    new_file = workspace_dir / "write_test.txt"

    result = write_tool._run(str(new_file), "New content")
    assert "success" in result.lower()
    assert new_file.exists()
    assert new_file.read_text() == "New content"


def test_write_file_tool_outside(test_environment):
    """Test WriteFileTool with files outside workspace."""
    write_tool = test_environment["write_tool"]
    outside_dir = test_environment["outside_dir"]
    outside_file = outside_dir / "write_outside.txt"

    result = write_tool._run(str(outside_file), "Should not be written")
    assert "access denied" in result.lower()
    assert not outside_file.exists()


def test_list_directory_workspace(test_environment):
    """Test ListDirectoryTool with directories in workspace."""
    list_tool = test_environment["list_tool"]
    workspace_dir = test_environment["workspace_dir"]

    result = list_tool._run(str(workspace_dir))
    assert "test_file.txt" in result


def test_list_directory_outside(test_environment):
    """Test ListDirectoryTool with directories outside workspace."""
    list_tool = test_environment["list_tool"]
    outside_dir = test_environment["outside_dir"]

    result = list_tool._run(str(outside_dir))
    assert "outside_file.txt" in result
