"""
Pydantic schemas for browser agent structured outputs.

These schemas are used with Ollama's `format` parameter to guarantee
the VLM returns valid, parseable JSON matching our action format.
"""

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Enumeration of available browser actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    DONE = "done"


class BrowserAction(BaseModel):
    """
    Schema for browser automation actions.
    
    This model is used with Ollama's structured output feature to ensure
    the VLM always returns valid JSON matching this exact structure.
    
    Usage with Ollama:
        response = ollama.chat(
            model="qwen3-vl:4b",
            messages=[...],
            format=BrowserAction.model_json_schema()
        )
        action = BrowserAction.model_validate_json(response.message.content)
    """
    action: Literal["navigate", "click", "type", "scroll", "wait", "done"] = Field(
        description="The type of browser action to perform"
    )
    target: str = Field(
        description="Target for the action: URL for navigate, element description for click/type, direction for scroll, seconds for wait, result summary for done"
    )
    value: Optional[str] = Field(
        default=None,
        description="Additional value for type actions (the text to type)"
    )
    reasoning: str = Field(
        description="Explanation of why this action was chosen"
    )
    
    @property
    def action_type(self) -> ActionType:
        """Get the action as an ActionType enum for compatibility."""
        return ActionType(self.action)
    
    def __str__(self) -> str:
        if self.value:
            return f"{self.action}({self.target}, value={self.value[:20]}...)"
        return f"{self.action}({self.target})"
