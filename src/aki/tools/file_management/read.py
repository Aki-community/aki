"""Tool for reading files from disk."""

import os
import json
import mimetypes
from pathlib import Path
from typing import Optional, Type, Dict, Any
import chardet
from markitdown import MarkItDown
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base_tools import ReadFileTool as ReadFileBase


class ReadFileInput(BaseModel):
    """Input for ReadFileTool."""

    file_path: str = Field(..., description="name of file")
    max_size: int = Field(
        default=1024 * 1024,  # 1 MB default
        description="Maximum file size to read in bytes. Default is 1 MB.",
    )
    truncate_lines: Optional[int] = Field(
        default=1000,  # Default to first 1000 lines
        description="Maximum number of lines to read. None means read all lines.",
    )
    convert_to_markdown: bool = Field(
        default=False,
        description="Convert non-text files (PDF, Word, Excel, PowerPoint, ZIP, etc.) to Markdown format",
    )


class ReadFileTool(ReadFileBase, BaseTool):
    """Tool that reads a file."""

    name: str = "read_file"
    args_schema: Type[BaseModel] = ReadFileInput
    description: str = "Read file from disk with size and line limitations"

    def _run(
        self,
        file_path: str,
        max_size: int = 1024 * 1024,  # 1 MB default
        truncate_lines: Optional[int] = 1000,
        convert_to_markdown: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Save original path for reporting
        original_path = file_path

        # Resolve the path
        resolved_path = self.resolve_path(file_path)

        # Prepare result structure
        result = {
            "original_path": original_path,
            "resolved_path": resolved_path,
            "content": None,
            "metadata": {
                "file_exists": False,
                "file_size": None,
                "line_count": None,
                "truncated_size": False,
                "truncated_lines": False,
                "mime_type": None,
                "encoding": "utf-8",  # Default assumption
            },
            "error": None,
        }

        # Check if we got an error message
        if resolved_path.startswith("Error:"):
            result["error"] = resolved_path
            return self._return_result(result, run_manager)

        try:
            path_obj = Path(resolved_path)

            # Check if file exists
            if not path_obj.exists():
                result["error"] = f"File does not exist: {file_path}"
                return self._return_result(result, run_manager)

            if not path_obj.is_file():
                result["error"] = f"Path is not a file: {file_path}"
                return self._return_result(result, run_manager)

            # Update metadata
            result["metadata"]["file_exists"] = True
            result["metadata"]["file_size"] = os.path.getsize(path_obj)

            # Try to guess mime type
            try:
                result["metadata"]["mime_type"] = mimetypes.guess_type(path_obj)[0]
            except:
                pass

            # Check file size
            file_size = result["metadata"]["file_size"]
            file_extension = path_obj.suffix.lower()

            # Check if we should attempt format conversion
            should_convert = convert_to_markdown and file_extension in [
                ".pdf",
                ".docx",
                ".doc",
                ".xlsx",
                ".xls",
                ".pptx",
                ".ppt",
                ".html",
                ".htm",
                ".zip",
            ]

            if should_convert:
                try:
                    md = MarkItDown()
                    # Use markitdown to convert the file
                    markdown_content = md.convert(str(path_obj)).text_content
                    content = markdown_content

                    # Add metadata about the conversion
                    result["metadata"]["converted_from"] = file_extension
                    result["metadata"]["conversion_successful"] = True

                    # Set content directly
                    result["content"] = content
                    return self._return_result(result, run_manager)

                except Exception as e:
                    # If conversion fails, fall back to regular file reading
                    result["metadata"]["conversion_error"] = str(e)

            # Regular file reading for text files
            # Detect file encoding if not binary
            if file_size > 0 and file_size <= max_size:
                try:
                    # Read up to 1MB of the file as binary
                    sample_size = min(file_size, 1024 * 1024)  # Max 1MB sample
                    with open(path_obj, "rb") as f:
                        raw_data = f.read(sample_size)

                    # Detect encoding
                    encoding_result = chardet.detect(raw_data)
                    if (
                        encoding_result["confidence"] > 0.7
                    ):  # Only use if confidence is high enough
                        result["metadata"]["encoding"] = encoding_result["encoding"]
                        result["metadata"]["encoding_confidence"] = encoding_result[
                            "confidence"
                        ]
                except Exception:
                    # If encoding detection fails, fallback to utf-8
                    pass

            # Get the detected encoding or default
            encoding = result["metadata"]["encoding"]

            try:
                if file_size > max_size:
                    # If file is too large, read only up to max_size
                    with path_obj.open("r", encoding=encoding) as f:
                        content = f.read(max_size)
                        result["metadata"]["truncated_size"] = True
                else:
                    # Read entire file
                    with path_obj.open("r", encoding=encoding) as f:
                        content = f.read()
            except UnicodeDecodeError:
                # Handle binary files
                result["error"] = "Cannot read binary file with text encoding"
                result["metadata"]["encoding"] = "binary"
                return self._return_result(result, run_manager)

            # Count and truncate lines if specified
            lines = content.splitlines()
            result["metadata"]["line_count"] = len(lines)

            if truncate_lines is not None and len(lines) > truncate_lines:
                content = "\n".join(lines[:truncate_lines])
                result["metadata"]["truncated_lines"] = True

            # Set content
            result["content"] = content

            return self._return_result(result, run_manager)

        except Exception as e:
            result["error"] = f"Error: {str(e)}"
            return self._return_result(result, run_manager)

    def _return_result(
        self,
        result: Dict[str, Any],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Format the result based on the requested format (JSON or text)."""
        # For backward compatibility with text format
        if (
            hasattr(run_manager, "is_text_format_requested")
            and run_manager.is_text_format_requested()
        ):
            return self._format_as_text(result)

        return json.dumps(result, indent=2)

    def _format_as_text(self, result: Dict[str, Any]) -> str:
        """Format the result as text for backward compatibility."""
        # If there's an error, return it directly
        if result["error"]:
            return result["error"]

        # Otherwise return the content
        if result["content"] is not None:
            content = result["content"]

            # Add truncation messages if needed
            if result["metadata"]["truncated_size"]:
                content += (
                    f"\n\n[TRUNCATED: File size {result['metadata']['file_size']} bytes exceeds "
                    f"maximum read size. Showing partial content.]"
                )

            if result["metadata"]["truncated_lines"]:
                content += (
                    f"\n\n[TRUNCATED: File contains {result['metadata']['line_count']} lines. "
                    f"Showing first {len(result['content'].splitlines())} lines.]"
                )

            return content

        return "Error: No content available"
