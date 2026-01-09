"""
Browser Agent Module - AI-powered browser automation.

This module provides a modular, well-architected browser automation agent
using Qwen3-VL vision model via Ollama with structured outputs.
"""

from core.agent.schemas import BrowserAction, ActionType
from core.agent.agent import BrowserAgent

__all__ = ["BrowserAgent", "BrowserAction", "ActionType"]
