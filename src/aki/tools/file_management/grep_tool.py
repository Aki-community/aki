import json
import subprocess
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from .base_tools import ReadFileTool as ReadFileBase


class GrepToolInput(BaseModel):
    """Input for GrepTool."""

    pattern: str = Field(
        ...,
        description="The regular expression pattern to search for",
    )
    path: str = Field(
        default=".",
        description="Directory to search in (defaults to current directory)",
    )
    glob: str = Field(
        default="",
        description="Glob pattern for filtering files (e.g., '*.py', '*.{ts,tsx}', 'src/**/*.js')",
    )
    max_results: int = Field(
        default=50, description="Maximum number of results to return"
    )
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )
    sort_by: str = Field(
        default="modified",
        description="How to sort results: 'modified' (most recent first), 'path' (alphabetically), or 'relevance'",
    )

    # Validate sort_by to ensure it's a valid option
    @field_validator("sort_by")
    def validate_sort_by(cls, v):
        valid_options = ["modified", "path", "relevance"]
        if v not in valid_options:
            raise ValueError(f"sort_by must be one of: {', '.join(valid_options)}")
        return v


class GrepTool(ReadFileBase, BaseTool):
    """Tool for searching file contents using regular expressions."""

    name: str = "grep"
    args_schema = GrepToolInput
    description: str = (
        """Searches file contents using regular expressions and returns matching lines.

Useful for finding code patterns, error messages, or specific text in files.

Examples:
- Find all error logging: grep pattern="log.*(Error|Exception)" glob="*.py"
- Find function definitions: grep pattern="function\\s+\\w+" glob="src/**/*.js"
- Find TODO comments: grep pattern="TODO:" path="src" glob="*.{ts,js,py}"
- Search specific files: grep pattern="config" glob="**/*.json"
- Search external directory: grep pattern="mcp" path="/Users/username/projects"

When searching directories outside the current workspace, provide an absolute path.
"""
    )

    def _run(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "",
        max_results: int = 50,
        case_sensitive: bool = False,
        sort_by: str = "modified",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the grep tool."""
        # Resolve path using the proper workspace-aware path resolver
        resolved_path = self.resolve_path(path)

        # Check if path resolution had errors
        if resolved_path.startswith("Error:"):
            return resolved_path

        # Prepare the result structure
        result = {
            "pattern": pattern,
            "path": path,
            "resolved_path": resolved_path,
            "glob": glob,
            "matches": [],
            "error": None,
        }

        try:
            # Build the ripgrep command
            cmd = ["rg"]

            # Add case sensitivity
            if not case_sensitive:
                cmd.append("-i")  # Case insensitive

            # Add line number flag
            cmd.append("-n")  # Include line numbers

            # Add glob pattern if provided
            if glob:
                cmd.extend(["--glob", glob])

            # Limit results if needed
            if max_results > 0:
                cmd.extend(["--max-count", str(max_results)])

            # Add sorting options
            if sort_by == "path":
                cmd.append("--sort=path")
            # Note: ripgrep doesn't have a direct modified-time sort,
            # but we'll keep the parameter for API compatibility

            # Add search pattern and path
            cmd.append(pattern)
            cmd.append(resolved_path)

            # Execute command
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # ripgrep returns exit code 1 when no matches found, which is not an error
            if process.returncode not in [0, 1]:
                result["error"] = (
                    f"Search failed with code {process.returncode}: {process.stderr}"
                )
                return json.dumps(result, indent=2)

            # Return the raw output
            return process.stdout or "No matches found"

        except Exception as e:
            result["error"] = f"Error executing search: {str(e)}"
            return json.dumps(result, indent=2)
