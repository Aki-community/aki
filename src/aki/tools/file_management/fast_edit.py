"""Tool for making targeted edits to files."""

import re
from typing import Optional, Type
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base_tools import WriteFileTool as WriteFileBase
from .file_paths import FilePathResolver, OPERATION_READ


class FastEditInput(BaseModel):
    """Input for FastEditTool."""

    file_path: str = Field(..., description="Path to the file to edit")
    patch_content: str = Field(
        ...,
        description="""Patch content in search/replace block format with these exact components:
                    1. Search block starting with <<<<<<< SEARCH
                    2. Original content to find
                    3. Divider line =======
                    4. Updated content to replace with
                    5. End marker >>>>>>> REPLACE
                    
                    Example:
                    <<<<<<< SEARCH
                    private String name;
                    private int age;
                    =======
                    private String name;
                    private int age;
                    private String email;
                    private String phone;
                    >>>>>>> REPLACE
                    
                    Note: Search content must match exactly with the target file
                    Multiple blocks can be provided for multiple changes""",
    )


class FastEditTool(WriteFileBase, BaseTool):
    """Tool for fast editing files using patch-style search/replace blocks."""

    name: str = "fast_edit_file"
    args_schema: Type[BaseModel] = FastEditInput
    description: str = """Fast edit a file using patch-style search/replace blocks.
    Useful for making targeted changes to specific parts of a file without rewriting the entire content.
    The patch must use the exact format with <<<<<<< SEARCH, =======, and >>>>>>> REPLACE markers.
    If edit failed, please read the file and try again with exact match content.
    """

    def parse_patches(self, patch_content: str):
        """Parse the patch content into search/replace pairs."""
        # Pattern to match the blocks
        pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        # Use re.DOTALL to make "." match newlines
        matches = re.finditer(pattern, patch_content, re.DOTALL)

        patches = []
        for match in matches:
            search_text = match.group(1)
            replace_text = match.group(2)
            patches.append((search_text, replace_text))

        return patches

    def _run(
        self,
        file_path: str,
        patch_content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # We need to read the file first (with read permissions)
        resolver = FilePathResolver(root_dir=self.root_dir)
        try:
            read_path = resolver.resolve_path(file_path, operation=OPERATION_READ)
        except Exception as e:
            return f"Error reading file: {str(e)}"

        # Check if the file is writable (with write permissions)
        write_result = self.resolve_path(file_path)
        if write_result.startswith("Error:"):
            return write_result

        try:
            path_obj = Path(read_path)

            # Read the original content
            with path_obj.open("r", encoding="utf-8") as f:
                original_content = f.read()

            # Parse the patches
            patches = self.parse_patches(patch_content)

            if not patches:
                return "Error: No valid patches found in the patch content. Make sure to use the correct format."

            # Apply the patches
            content = original_content
            applied_patches = 0

            for search_text, replace_text in patches:
                if search_text in content:
                    content = content.replace(search_text, replace_text)
                    applied_patches += 1
                else:
                    return (
                        f"Error editing file: Could not find the exact text to replace.\n"
                        f"Make sure your SEARCH block matches exactly with the content in the file.\n"
                        f"Applied {applied_patches}/{len(patches)} patches successfully."
                    )

            # Write the modified content
            with path_obj.open("w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully applied patches to {file_path}"

        except Exception as e:
            return f"Error editing file: {str(e)}"
