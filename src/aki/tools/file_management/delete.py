"""Tool for deleting files."""

import os
from pathlib import Path
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base_tools import WriteFileTool as WriteFileBase


class FileDeleteInput(BaseModel):
    """Input for DeleteFileTool."""

    file_path: str = Field(..., description="Path of the file to delete")


class DeleteFileTool(WriteFileBase, BaseTool):  # type: ignore[override, override]
    """Tool that deletes a file."""

    name: str = "file_delete"
    args_schema: Type[BaseModel] = FileDeleteInput
    description: str = "Delete a file"

    def _run(
        self,
        file_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        resolved_path = self.resolve_path(file_path)

        # Check if we got an error message
        if resolved_path.startswith("Error:"):
            return resolved_path

        try:
            path_obj = Path(resolved_path)

            if not path_obj.exists():
                return f"Error: No such file or directory: {file_path}"

            os.remove(path_obj)
            return f"File deleted successfully: {file_path}."

        except Exception as e:
            return f"Error: {str(e)}"
