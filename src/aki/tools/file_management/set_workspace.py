import os
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import chainlit as cl


class SetWorkspaceInput(BaseModel):
    """Input for SetWorkspaceTool."""

    path: str = Field(
        ...,
        description="The absolute or relative path to set as the workspace directory. Must be a valid, existing directory.",
    )


class SetWorkspaceTool(BaseTool):
    """Tool that sets the workspace directory for file operations."""

    name: str = "set_workspace"
    args_schema: Type[BaseModel] = SetWorkspaceInput
    description: str = """Set the workspace directory path where all write operations (create/edit/delete files) will be restricted to.
    This is a security measure to ensure file operations only occur within the designated workspace."""

    def _run(
        self,
        path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not os.path.isdir(path):
            raise ValueError(
                f"Invalid workspace path: {path}. The path must be an existing directory."
            )

        state = cl.user_session.get("state")
        path = os.path.abspath(path)
        if os.path.exists(path):
            os.chdir(path)
            state["workspace_dir"] = path
            return f"Workspace successfully set to: {state['workspace_dir']}"
        else:
            return f"Workspace {path} does not exists"

    async def _arun(
        self,
        path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async implementation of the tool."""
        return self._run(path=path, run_manager=run_manager)


def create_set_workspace_tool() -> BaseTool:
    """Create and return the set workspace tool."""
    return SetWorkspaceTool()
