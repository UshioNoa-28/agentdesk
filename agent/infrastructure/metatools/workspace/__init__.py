"""Workspace Meta Tools.

Each operation lives in its own module; this package keeps the historical
``agent.infrastructure.metatools.workspace`` import path as a small compatibility facade.
"""

# ``os`` is intentionally imported here as a compatibility alias.  Existing
# callers/tests that patch ``agent.infrastructure.metatools.workspace.os.replace`` still patch
# the shared ``os`` module used by ``edit_file``.
import os  # noqa: F401

from agent.infrastructure.metatools.workspace._base import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_SEARCH_RESULTS,
    UserPathError,  # noqa: F401
)
from agent.infrastructure.metatools.workspace.edit_file import EDIT_FILE_DEFINITION, EditFileTool
from agent.infrastructure.metatools.workspace.read_file import READ_FILE_DEFINITION, ReadFileTool
from agent.infrastructure.metatools.workspace.search_text import (
    SEARCH_TEXT_DEFINITION,
    SearchTextTool,
)

__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_SEARCH_RESULTS",
    "EDIT_FILE_DEFINITION",
    "READ_FILE_DEFINITION",
    "SEARCH_TEXT_DEFINITION",
    "EditFileTool",
    "ReadFileTool",
    "SearchTextTool",
]
