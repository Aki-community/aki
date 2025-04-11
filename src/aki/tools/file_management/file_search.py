import os
import logging
import json
from typing import Optional, Type, List, Dict, Any
from multiprocessing import Pool
import pathspec
from wcmatch import glob

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base_tools import ReadFileTool as ReadFileBase


class FileSearchInput(BaseModel):
    """Input for FileSearchTool."""

    dir_path: str = Field(
        default=".",
        description="Subdirectory to search in.",
    )
    pattern: str = Field(
        ...,
        description="Unix shell regex for filename matching, where * matches everything.",
    )
    keyword: str = Field(
        default="",
        description="Optional keyword to search within file contents.",
    )
    max_results: int = Field(
        default=100, description="Maximum number of search results to return."
    )
    include_hidden: bool = Field(
        default=False, description="Include files/directories starting with '.'"
    )
    respect_gitignore: bool = Field(
        default=True, description="Exclude files matched by .gitignore patterns"
    )
    recursive: bool = Field(
        default=True,
        description="Search recursively in subdirectories (automatically adds **/ to pattern if not present)",
    )


def search_file_content(args: tuple) -> Optional[str]:
    """Search for keyword in a file. Used with multiprocessing."""
    filepath, keyword = args
    try:
        # First check if file is binary
        with open(filepath, "rb") as file:
            chunk = file.read(1024)
            if b"\x00" in chunk:  # Simple binary check
                return None

        # If not binary, search for keyword
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            if keyword.lower() in content.lower():
                return filepath
    except (UnicodeDecodeError, IOError):
        pass
    return None


class FileSearchTool(ReadFileBase, BaseTool):
    """Tool that searches for files by name pattern and/or content keywords."""

    name: str = "file_search"
    args_schema: Type[BaseModel] = FileSearchInput
    description: str = (
        """Search for files that match the filename pattern and/or contain specific keywords.

IMPORTANT DEFAULTS AND BEHAVIORS:
1. Recursive Search:
   - Recursive subdirectory search is ENABLED by default
   - Pattern is automatically prefixed with **/ if not present
   - Use recursive=False to search only in current directory
   - Example: "*.py" becomes "**/*.py" by default

2. Hidden Files:
   - Files/directories starting with '.' are EXCLUDED by default
   - Use include_hidden=True to include hidden files
   - Example: .git/, .config/, .env files are hidden

3. GitIgnore Rules:
   - Files matched by .gitignore patterns are EXCLUDED by default
   - Use respect_gitignore=False to ignore .gitignore rules
   - Common excludes: node_modules/, __pycache__/, build/

4. Path Handling:
   - Relative paths are resolved from dir_path
   - Absolute paths are respected if provided
   - Forward slashes (/) work on all platforms

SEARCH PATTERNS:

1. Simple File Search:
   pattern="config.json"        # Find specific file
   pattern="*.py"              # All Python files
   pattern="*.{js,ts}"         # All JavaScript and TypeScript files

2. Directory-Specific Search:
   pattern="src/*.py"          # Python files in src/ directory only
   pattern="test/**/*.py"      # Python files in test/ and subdirectories
   pattern="**/tests/*.py"     # Python files in any tests directory

3. Multiple Pattern Search:
   pattern="*.py|*.java"       # Python OR Java files
   pattern="src/*.ts|libs/*.ts" # TypeScript files in src/ OR libs/
   pattern="test_*.py|*_test.py" # Files starting with test_ OR ending with _test.py

4. Hidden File Search:
   pattern=".env", include_hidden=True           # Find .env files
   pattern=".git/**", include_hidden=True        # Search in .git directory
   pattern="**/.config/*", include_hidden=True   # Find files in .config directories

5. Content Search:
   pattern="*.py", keyword="TODO"     # Python files containing "TODO"
   pattern="**/*.md", keyword="FIXME" # Markdown files containing "FIXME"
   pattern="*.{js,ts}", keyword="DEBUG" # JS/TS files containing "DEBUG"

TIPS:
- Use | to combine multiple patterns
- Use ** for recursive directory search
- Use *.{ext1,ext2} for multiple extensions
- Always use include_hidden=True when searching for:
  * Configuration files (.env, .config)
  * Git files (.git, .gitignore)
  * Hidden directories (.vscode, .idea)
- Use respect_gitignore=False when searching in:
  * Build directories
  * Node modules
  * Cache directories
"""
    )

    def _should_include_file(self, filename: str, include_hidden: bool) -> bool:
        """Determine if a file should be included based on hidden status."""
        return include_hidden or not filename.startswith(".")

    def _load_gitignore(self, dir_path: str) -> Optional[pathspec.PathSpec]:
        """Load .gitignore patterns if the file exists."""
        gitignore_path = os.path.join(dir_path, ".gitignore")
        try:
            if os.path.isfile(gitignore_path):
                with open(gitignore_path, "r") as f:
                    patterns = f.read().splitlines()
                return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except Exception:
            # If there's any error reading .gitignore, return None
            pass
        return None

    def _normalize_path(self, path: str) -> str:
        """Normalize path to absolute path with forward slashes."""
        return os.path.abspath(path).replace("\\", "/")

    def _is_path_hidden(self, path: str) -> bool:
        """Check if any component of the path is hidden (starts with .)"""
        return any(part.startswith(".") for part in path.split("/") if part)

    def _get_matching_files(
        self,
        dir_path: str,
        pattern: str,
        include_hidden: bool,
        respect_gitignore: bool = True,
        max_results: Optional[int] = None,
    ) -> List[str]:
        """Get list of files matching the pattern using wcmatch glob.

        Args:
            dir_path: Base directory for search
            pattern: File pattern to match
            include_hidden: Whether to include hidden files
            respect_gitignore: Whether to respect .gitignore rules
            max_results: Maximum number of results to return

        Returns:
            List of matching file paths relative to dir_path

        Raises:
            FileNotFoundError: If dir_path doesn't exist
            NotADirectoryError: If dir_path is not a directory
            ValueError: If pattern is invalid
        """
        # Normalize directory path
        dir_path = self._normalize_path(dir_path)

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Load gitignore if needed
        gitignore = self._load_gitignore(dir_path) if respect_gitignore else None

        # Handle multiple patterns (pipe-separated)
        patterns = [p.strip() for p in pattern.split("|")]

        matching_files = []
        try:
            # Set up glob flags
            flags = glob.GLOBSTAR | glob.BRACE
            if include_hidden:
                flags |= glob.DOTGLOB

            for p in patterns:
                # Handle absolute patterns
                if os.path.isabs(p):
                    search_pattern = self._normalize_path(p)
                else:
                    # Handle patterns with and without **
                    if "**" in p:
                        search_pattern = os.path.join(dir_path, p)
                    else:
                        # Don't automatically add ** for non-recursive patterns
                        search_pattern = os.path.join(dir_path, p)

                try:
                    matches = glob.glob(search_pattern, flags=flags)
                    for match in matches:
                        # Normalize and convert to relative path
                        match = self._normalize_path(match)
                        rel_path = os.path.relpath(match, dir_path).replace("\\", "/")

                        # Skip hidden paths if not included
                        if not include_hidden and self._is_path_hidden(rel_path):
                            continue

                        # Check gitignore patterns
                        if gitignore is not None and gitignore.match_file(rel_path):
                            continue

                        if rel_path not in matching_files:
                            matching_files.append(rel_path)
                            if (
                                max_results is not None
                                and len(matching_files) >= max_results
                            ):
                                return matching_files
                except glob.WcMatchError as e:
                    raise ValueError(f"Invalid pattern '{p}': {str(e)}")

        except Exception as e:
            raise ValueError(f"Error processing pattern: {str(e)}")

        return matching_files

    def _run(
        self,
        pattern: str,
        dir_path: str = ".",
        keyword: str = "",
        max_results: int = 100,
        include_hidden: bool = False,
        respect_gitignore: bool = True,
        recursive: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Save the original path and pattern for reporting
        original_path = dir_path
        original_pattern = pattern

        # Prepare the result structure
        result = {
            "original_path": original_path,
            "original_pattern": original_pattern,
            "resolved_path": None,
            "recursive": recursive,
            "include_hidden": include_hidden,
            "respect_gitignore": respect_gitignore,
            "keyword": keyword,
            "matches": [],
            "total_matches": 0,
            "error": None,
        }

        # Resolve directory path
        resolved_dir_path = self.resolve_path(dir_path)
        result["resolved_path"] = resolved_dir_path

        # Check if path resolution had errors
        if resolved_dir_path.startswith("Error:"):
            result["error"] = resolved_dir_path
            return json.dumps(result, indent=2)

        try:
            # Check if searching for hidden files without include_hidden
            has_hidden_pattern = any(
                part.startswith(".") for part in pattern.replace("\\", "/").split("/")
            )
            if has_hidden_pattern and not include_hidden:
                logging.warning(
                    "Pattern includes hidden files/directories but include_hidden=False. "
                    "Hidden files will be excluded."
                )

            # Add ** for recursive search if not present and recursive is True
            if recursive and "**" not in pattern:
                pattern = f"**/{pattern}"

            # Get files matching the pattern
            try:
                matching_files = self._get_matching_files(
                    resolved_dir_path,
                    pattern,
                    include_hidden,
                    respect_gitignore,
                    (
                        None if keyword else max_results
                    ),  # Only apply max_results here if no keyword search
                )
            except ValueError as e:
                result["error"] = str(e)
                return json.dumps(result, indent=2)

            matches = []
            # If keyword is provided, use multiprocessing to search file contents
            if keyword:
                with Pool() as pool:
                    search_args = [
                        (os.path.join(resolved_dir_path, f), keyword)
                        for f in matching_files
                    ]
                    # Collect all matches first to get accurate count
                    all_matches = []
                    for result_path in pool.imap_unordered(
                        search_file_content, search_args
                    ):
                        if result_path:
                            relative_path = os.path.relpath(
                                result_path, resolved_dir_path
                            )
                            all_matches.append(relative_path)

                    # Record total count and truncate to max_results
                    result["total_matches"] = len(all_matches)

                    # Convert matches to detailed format
                    for i, match_path in enumerate(all_matches):
                        if i >= max_results:
                            break

                        abs_path = os.path.join(resolved_dir_path, match_path)
                        try:
                            file_size = os.path.getsize(abs_path)
                            file_ext = os.path.splitext(match_path)[1].lower()
                        except (OSError, IOError):
                            file_size = None
                            file_ext = None

                        matches.append(
                            {
                                "path": match_path,
                                "absolute_path": abs_path,
                                "extension": file_ext,
                                "size": file_size,
                                "contains_keyword": True,
                            }
                        )
            else:
                # Record total count
                result["total_matches"] = len(matching_files)

                # Convert matches to detailed format (limit to max_results)
                for i, match_path in enumerate(matching_files):
                    if i >= max_results:
                        break

                    abs_path = os.path.join(resolved_dir_path, match_path)
                    try:
                        file_size = os.path.getsize(abs_path)
                        file_ext = os.path.splitext(match_path)[1].lower()
                    except (OSError, IOError):
                        file_size = None
                        file_ext = None

                    matches.append(
                        {
                            "path": match_path,
                            "absolute_path": abs_path,
                            "extension": file_ext,
                            "size": file_size,
                        }
                    )

            # Add matches to the result
            result["matches"] = matches

            # Return JSON result
            json_result = json.dumps(result, indent=2)

            # For backward compatibility, check if text format is requested
            if (
                hasattr(run_manager, "is_text_format_requested")
                and run_manager.is_text_format_requested()
            ):
                return self._format_as_text(result)

            return json_result

        except FileNotFoundError as e:
            result["error"] = str(e)
            return json.dumps(result, indent=2)
        except NotADirectoryError as e:
            result["error"] = str(e)
            return json.dumps(result, indent=2)
        except Exception as e:
            result["error"] = str(e)
            return json.dumps(result, indent=2)

    def _format_as_text(self, result_dict: Dict[str, Any]) -> str:
        """Format the JSON result as text for backward compatibility."""
        lines = []

        # Check for errors
        if result_dict.get("error"):
            return result_dict["error"]

        # If we have matches, list them
        if result_dict["matches"]:
            for match in result_dict["matches"]:
                lines.append(match["path"])

            # Add truncation message if needed
            if result_dict["total_matches"] > len(result_dict["matches"]):
                lines.append(
                    f"... (showing first {len(result_dict['matches'])} of {result_dict['total_matches']} results)"
                )
        else:
            # Build helpful error message for no matches
            search_type = "pattern and keyword" if result_dict["keyword"] else "pattern"
            lines.append(
                f"No files found matching {search_type} '{result_dict['original_pattern']}' in directory '{result_dict['original_path']}'"
            )

            has_hidden_pattern = any(
                part.startswith(".")
                for part in result_dict["original_pattern"]
                .replace("\\", "/")
                .split("/")
            )

            if has_hidden_pattern and not result_dict["include_hidden"]:
                lines.append(
                    "Note: Pattern includes hidden files but include_hidden=False"
                )
                lines.append(
                    "      Use include_hidden=True to include hidden files/directories"
                )

            if result_dict["respect_gitignore"]:
                lines.append("Note: Files matched by .gitignore are excluded")
                lines.append("      Use respect_gitignore=False to include them")

        return "\n".join(lines)
