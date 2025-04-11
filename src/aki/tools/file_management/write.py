"""Tool for writing files to disk."""

import json
from pathlib import Path
from typing import Optional, Type, Dict, Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base_tools import WriteFileTool as WriteFileBase


class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="name of file")
    text: str = Field(..., description="text to write to file")
    append: bool = Field(
        default=False, description="Whether to append to an existing file."
    )


class WriteFileTool(WriteFileBase, BaseTool):
    """Tool that writes a file to disk."""

    name: str = "write_file"
    args_schema: Type[BaseModel] = WriteFileInput
    description: str = "Write file to disk"

    def _run(
        self,
        file_path: str,
        text: str,
        append: bool = False,
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
            "success": False,
            "append_mode": append,
            "characters_written": 0,
            "error": None,
        }

        # Check if we got an error message
        if resolved_path.startswith("Error:"):
            result["error"] = resolved_path
            return self._return_result(result, run_manager)

        try:
            path_obj = Path(resolved_path)

            # Ensure parent directory exists
            path_obj.parent.mkdir(exist_ok=True, parents=False)

            # Write or append to file
            mode = "a" if append else "w"
            with path_obj.open(mode, encoding="utf-8") as f:
                f.write(text)

            # Update result
            result["success"] = True
            result["characters_written"] = len(text)

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
        if result["error"]:
            return result["error"]

        if result["success"]:
            return f"File written successfully to {result['original_path']}."
        else:
            return f"Failed to write to {result['original_path']}."
