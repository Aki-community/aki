"""Dir listing with gitignore."""

import os
import pathspec
import json
from dataclasses import dataclass
from typing import Optional, Type, Dict, List, Tuple, Any
from pathlib import Path
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from .base_tools import ReadFileTool as ReadFileBase


@dataclass
class DirectoryStats:
    total_files: int = 0
    total_dirs: int = 0
    file_types: Dict[str, int] = None
    total_size: int = 0

    def __post_init__(self):
        if self.file_types is None:
            self.file_types = {}

    def add_file(self, path: Path) -> None:
        self.total_files += 1
        ext = path.suffix.lower() or "(no extension)"
        self.file_types[ext] = self.file_types.get(ext, 0) + 1
        try:
            self.total_size += path.stat().st_size
        except (OSError, IOError):
            pass

    def add_dir(self) -> None:
        self.total_dirs += 1

    def format_size(self) -> str:
        size = self.total_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def get_summary(self) -> str:
        summary = [
            f"Total: {self.total_files} files, {self.total_dirs} directories",
            f"Size: {self.format_size()}",
        ]
        if self.file_types:
            type_summary = sorted(
                [(ext, count) for ext, count in self.file_types.items()],
                key=lambda x: (-x[1], x[0]),
            )
            summary.append("File types:")
            for ext, count in type_summary[:5]:
                summary.append(f"  {ext}: {count} files")
            if len(type_summary) > 5:
                summary.append(f"  ... and {len(type_summary)-5} more types")
        return "\n".join(summary)

    def get_summary_dict(self) -> Dict[str, Any]:
        stats = {
            "files": self.total_files,
            "dirs": self.total_dirs,
            "size": self.format_size(),
        }

        return stats


class DirectoryListingInput(BaseModel):
    dir_path: str = Field(default=".", description="Subdirectory to list.")
    max_results: int = Field(
        default=200, description="Maximum number of entries to return."
    )
    include_hidden: bool = Field(
        default=False, description="Include files/directories starting with '.'"
    )
    max_depth: int = Field(
        default=10,
        description="Maximum depth for recursive listing. Use -1 for unlimited.",
    )
    summarize: bool = Field(
        default=False,
        description="Summarize large directories instead of listing all files.",
    )
    respect_gitignore: bool = Field(
        default=True, description="Respect .gitignore patterns when listing files."
    )


class ListDirectoryTool(ReadFileBase, BaseTool):

    name: str = "list_directory"
    args_schema: Type[BaseModel] = DirectoryListingInput
    description: str = "List files and directories in a specified folder"

    def _should_summarize(self, path: Path, entry_count: int, max_results: int) -> bool:
        return entry_count > max_results // 2 or entry_count > 20

    def _load_gitignore(self, dir_path: str) -> Optional[pathspec.PathSpec]:
        gitignore_path = os.path.join(dir_path, ".gitignore")
        try:
            if os.path.isfile(gitignore_path):
                with open(gitignore_path, "r") as f:
                    patterns = f.read().splitlines()

                common = [
                    "__pycache__/",
                    "*.py[cod]",
                    "build/",
                    "dist/",
                    "node_modules/",
                    ".DS_Store",
                ]

                for pattern in common:
                    if pattern not in patterns:
                        patterns.append(pattern)

                return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except Exception:
            pass
        return None

    def _is_ignored(
        self,
        path: Path,
        root_path: str,
        gitignore_spec: Optional[pathspec.PathSpec],
        include_hidden: bool,
    ) -> bool:
        if not include_hidden and path.name.startswith("."):
            return True
        if gitignore_spec is not None:
            return gitignore_spec.match_file(os.path.relpath(path, root_path))
        return False

    def _list_directory_json(
        self,
        path: Path,
        root_path: str,
        gitignore_spec: Optional[pathspec.PathSpec],
        include_hidden: bool,
        max_results: int,
        max_depth: int,
        summarize: bool,
        current_depth: int = 0,
    ) -> Tuple[List[Dict[str, Any]], DirectoryStats]:
        try:
            entries = []
            stats = DirectoryStats()
            dir_entries = sorted(
                path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            )
            filtered_entries = [
                e
                for e in dir_entries
                if not self._is_ignored(e, root_path, gitignore_spec, include_hidden)
            ]
            should_summarize = summarize and self._should_summarize(
                path, len(filtered_entries), max_results
            )
            dirs = [e for e in filtered_entries if e.is_dir()]
            files = [e for e in filtered_entries if not e.is_dir()]
            entry_count = 0

            # Process directories
            for entry in dirs:
                if entry_count >= max_results:
                    entries.append({"type": "dir", "name": "...", "truncated": True})
                    return entries, stats

                try:
                    stats.add_dir()
                    dir_entry = {"type": "dir", "name": entry.name}
                    entry_count += 1

                    # Only show contents if depth allows
                    if not should_summarize and max_depth > 0:
                        sub_entries, sub_stats = self._list_directory_json(
                            entry,
                            root_path,
                            gitignore_spec,
                            include_hidden,
                            max_results - entry_count,
                            max_depth - 1,
                            summarize,
                            current_depth + 1,
                        )
                        if sub_entries:  # Only add children if there are any
                            dir_entry["children"] = sub_entries
                        stats.total_files += sub_stats.total_files
                        stats.total_dirs += sub_stats.total_dirs
                        stats.total_size += sub_stats.total_size
                        for ext, count in sub_stats.file_types.items():
                            stats.file_types[ext] = stats.file_types.get(ext, 0) + count

                    entries.append(dir_entry)
                except (OSError, IOError):
                    entries.append({"type": "dir", "name": entry.name, "error": True})

            # Process files
            if max_depth >= 0:  # Only show files if we haven't exceeded max_depth
                for entry in files:
                    if entry_count >= max_results:
                        entries.append(
                            {"type": "file", "name": "...", "truncated": True}
                        )
                        return entries, stats

                    try:
                        stats.add_file(entry)
                        ext = entry.suffix.lower()
                        entries.append(
                            {
                                "name": entry.name,
                            }
                        )
                        entry_count += 1
                    except (OSError, IOError):
                        entries.append(
                            {"type": "file", "name": entry.name, "error": True}
                        )

            # Simplified summary
            if should_summarize:
                sub_stats = DirectoryStats()
                for entry in filtered_entries:
                    try:
                        if entry.is_dir():
                            _, dir_stats = self._list_directory_json(
                                entry,
                                root_path,
                                gitignore_spec,
                                include_hidden,
                                max_results,
                                0,
                                False,
                                0,
                            )
                            sub_stats.total_files += dir_stats.total_files
                            sub_stats.total_dirs += dir_stats.total_dirs
                            sub_stats.total_size += dir_stats.total_size
                            for ext, count in dir_stats.file_types.items():
                                sub_stats.file_types[ext] = (
                                    sub_stats.file_types.get(ext, 0) + count
                                )
                        else:
                            sub_stats.add_file(entry)
                    except (OSError, IOError):
                        continue

                entries.append(
                    {"sum": f"{sub_stats.total_files}f/{sub_stats.total_dirs}d"}
                )

            return entries, stats

        except Exception as e:
            return [
                {"type": "error", "message": f"Error reading directory: {str(e)}"}
            ], DirectoryStats()

    def _run(
        self,
        dir_path: str = ".",
        max_results: int = 100,
        include_hidden: bool = False,
        max_depth: int = 3,
        summarize: bool = True,
        respect_gitignore: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Save the original path for reporting
        original_path = dir_path

        # Resolve the path
        resolved_path = self.resolve_path(dir_path)

        # Prepare the result structure
        result = {"path": original_path, "items": [], "stats": {}, "error": None}

        # Check if we got an error message
        if resolved_path.startswith("Error:"):
            result["error"] = resolved_path
            return json.dumps(result)

        try:
            path = Path(resolved_path)
            if not path.exists():
                result["error"] = f"Directory not found: {dir_path}"
                return json.dumps(result)

            if not path.is_dir():
                result["error"] = f"Not a directory: {dir_path}"
                return json.dumps(result)

            # Load gitignore if requested
            gitignore_spec = None
            if respect_gitignore:
                gitignore_spec = self._load_gitignore(str(path))

            # Generate JSON listing
            contents, stats = self._list_directory_json(
                path,
                path,
                gitignore_spec,
                include_hidden,
                max_results,
                max_depth,
                summarize,
                0,
            )

            result["items"] = contents
            result["stats"] = stats.get_summary_dict()

            return json.dumps(result)

        except Exception as e:
            result["error"] = f"Error: {str(e)}"
            return json.dumps(result)

    def _list_directory(
        self,
        path: Path,
        root_path: str,
        gitignore_spec: Optional[pathspec.PathSpec],
        include_hidden: bool,
        max_results: int,
        max_depth: int,
        summarize: bool,
        current_depth: int = 0,
    ) -> Tuple[List[str], DirectoryStats]:
        try:
            entries = []
            stats = DirectoryStats()
            indent = "  " * current_depth

            # Get directory contents
            dir_entries = sorted(
                path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            )

            # Filter entries based on hidden status and gitignore
            filtered_entries = [
                e
                for e in dir_entries
                if not self._is_ignored(e, root_path, gitignore_spec, include_hidden)
            ]

            # Check if we should summarize
            should_summarize = summarize and self._should_summarize(
                path, len(filtered_entries), max_results
            )

            # Process directories first, then files
            dirs = [e for e in filtered_entries if e.is_dir()]
            files = [e for e in filtered_entries if not e.is_dir()]

            entry_count = 0

            # Process directories
            for entry in dirs:
                if entry_count >= max_results:
                    entries.append(f"{indent}... (max results limit reached)")
                    return entries, stats

                try:
                    stats.add_dir()
                    entries.append(f"{indent}📁 {entry.name}/")
                    entry_count += 1

                    # Only show contents if depth allows
                    if not should_summarize and max_depth > 0:
                        sub_entries, sub_stats = self._list_directory(
                            entry,
                            root_path,
                            gitignore_spec,
                            include_hidden,
                            max_results - entry_count,
                            max_depth - 1,
                            summarize,
                            current_depth + 1,
                        )
                        entries.extend(sub_entries)
                        stats.total_files += sub_stats.total_files
                        stats.total_dirs += sub_stats.total_dirs
                        stats.total_size += sub_stats.total_size
                        for ext, count in sub_stats.file_types.items():
                            stats.file_types[ext] = stats.file_types.get(ext, 0) + count
                except (OSError, IOError):
                    entries.append(f"{indent}❌ {entry.name} (access error)")

            # Process files
            if max_depth >= 0:  # Only show files if we haven't exceeded max_depth
                for entry in files:
                    if entry_count >= max_results:
                        entries.append(f"{indent}... (max results limit reached)")
                        return entries, stats

                    try:
                        stats.add_file(entry)
                        # Add file icon based on type
                        if entry.suffix.lower() in [
                            ".py",
                            ".js",
                            ".java",
                            ".ts",
                            ".go",
                            ".c",
                            ".cpp",
                        ]:
                            icon = "📜"  # Script/code file
                        elif entry.suffix.lower() in [
                            ".md",
                            ".txt",
                            ".rst",
                            ".json",
                            ".yaml",
                            ".yml",
                        ]:
                            icon = "📝"  # Document/text file
                        elif entry.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif"]:
                            icon = "🖼️"  # Image file
                        else:
                            icon = "📄"  # Generic file

                        entries.append(f"{indent}{icon} {entry.name}")
                        entry_count += 1
                    except (OSError, IOError):
                        entries.append(f"{indent}❌ {entry.name} (access error)")

            # Add summary for large directories
            if should_summarize:
                sub_stats = DirectoryStats()
                for entry in filtered_entries:
                    try:
                        if entry.is_dir():
                            # Use a depth of 0 to avoid recursion for statistics
                            _, dir_stats = self._list_directory(
                                entry,
                                root_path,
                                gitignore_spec,
                                include_hidden,
                                max_results,
                                0,
                                False,
                                0,
                            )
                            sub_stats.total_files += dir_stats.total_files
                            sub_stats.total_dirs += dir_stats.total_dirs
                            sub_stats.total_size += dir_stats.total_size
                            for ext, count in dir_stats.file_types.items():
                                sub_stats.file_types[ext] = (
                                    sub_stats.file_types.get(ext, 0) + count
                                )
                        else:
                            sub_stats.add_file(entry)
                    except (OSError, IOError):
                        continue

                summary_lines = sub_stats.get_summary().split("\n")
                entries.extend([f"{indent}  {line}" for line in summary_lines])

            return entries, stats

        except Exception as e:
            return [f"Error reading directory: {str(e)}"], DirectoryStats()
