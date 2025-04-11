"""Tool for copying files."""

import shutil
from pathlib import Path
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base_tools import WriteFileTool as WriteFileBase
from .file_paths import FilePathResolver, OPERATION_READ


class CopyFileInput(BaseModel):
    """Input for CopyFileTool."""

    source_path: str = Field(..., description="Path of the file to copy")
    destination_path: str = Field(..., description="Path to save the copied file")


class CopyFileTool(WriteFileBase, BaseTool):
    """Tool that copies a file from source to destination."""

    name: str = "copy_file"
    args_schema: Type[BaseModel] = CopyFileInput
    description: str = "Create a copy of a file in a specified location"

    def _run(
        self,
        source_path: str,
        destination_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # For source path, we use READ operation since we're just reading from it
        resolver = FilePathResolver(root_dir=self.root_dir)
        try:
            source_resolved = resolver.resolve_path(
                source_path, operation=OPERATION_READ
            )
        except Exception as e:
            return f"Error with source file: {str(e)}"

        # For destination path, we use the write permission check
        dest_resolved = self.resolve_path(destination_path)
        if dest_resolved.startswith("Error:"):
            return dest_resolved

        try:
            source_obj = Path(source_resolved)
            dest_obj = Path(dest_resolved)

            if not source_obj.exists():
                return f"Error: Source file does not exist: {source_path}"

            # Ensure parent directory exists
            dest_obj.parent.mkdir(exist_ok=True, parents=False)

            # Copy the file
            shutil.copy2(source_obj, dest_obj)
            return f"File copied successfully from {source_path} to {destination_path}."

        except Exception as e:
            return f"Error: {str(e)}"
