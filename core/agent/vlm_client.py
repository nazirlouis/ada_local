"""
VLM Client - Ollama API client for vision-language model interactions.

Handles communication with Ollama using structured outputs and streaming
for thinking tokens.
"""

import json
import sys
from typing import Callable, Optional

import requests
from pydantic import ValidationError

from config import OLLAMA_URL, GRAY, RESET
from core.model_manager import ensure_exclusive_qwen
from core.agent.schemas import BrowserAction


# Configuration
VLM_MODEL = "qwen3-vl:4b"
VLM_TIMEOUT = 120
VLM_MAX_RETRIES = 2
DEBUG = False  # Set to True for verbose debug output


def get_action_schema() -> dict:
    """Get the JSON schema for BrowserAction."""
    return BrowserAction.model_json_schema()


class VLMClient:
    """
    Client for interacting with Ollama vision-language models.
    
    Uses structured outputs to guarantee valid JSON responses and
    streams thinking tokens for real-time UI feedback.
    """
    
    def __init__(
        self,
        model: str = VLM_MODEL,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the VLM client.
        
        Args:
            model: Ollama model name (e.g., "qwen3-vl:4b")
            on_thinking: Callback for thinking token streaming
            on_status: Callback for status updates
        """
        self.model = model
        self.on_thinking = on_thinking
        self.on_status = on_status
        self.ollama_url = f"{OLLAMA_URL}/chat"
    
    def _emit_status(self, text: str):
        """Emit status update."""
        if self.on_status:
            self.on_status(text)
        if DEBUG:
            print(f"{GRAY}[VLM] {text}{RESET}")
    
    def _emit_thinking(self, text: str):
        """Emit thinking token."""
        if self.on_thinking:
            self.on_thinking(text)
    
    def _build_prompt(self, task: str, step: int, max_steps: int, history: list[str] = None) -> str:
        """
        Build the system prompt for the VLM.
        
        Args:
            task: The user's task description
            step: Current step number
            max_steps: Maximum allowed steps
            history: Optional list of previous actions taken
        """
        history_text = ""
        if history:
            history_text = "\n\nPrevious actions taken:\n" + "\n".join(f"- {h}" for h in history[-5:])
        
        return f"""You are a browser automation agent. Your task is: {task}

Current step: {step}/{max_steps}{history_text}

Look at this screenshot of a web browser and decide what action to take next.

Available actions:
- navigate: Go to a URL (target = the URL)
- click: Click an element (target = element description like "the Search button")
- type: Type text into a field (target = field description, value = text to type)
- scroll: Scroll the page (target = "up" or "down")
- wait: Wait for page to load (target = seconds to wait, e.g., "1")
- done: Task is complete (target = result summary)

IMPORTANT:
- For click and type, describe the element using its label, placeholder, or visual appearance
- For type actions, ALWAYS specify both target (the field) and value (the text)
- When typing, include the ENTIRE phrase you want to type in the value field at once
- If the task is complete, use "done"
- Avoid repeating failed actions
- ALWAYS use DuckDuckGo (https://duckduckgo.com) for web searches, NOT Google"""
    
    async def get_action(
        self,
        screenshot_bytes: bytes,
        task: str,
        step: int,
        max_steps: int,
        history: list[str] = None
    ) -> BrowserAction:
        """
        Send screenshot to VLM and get the next action.
        
        Args:
            screenshot_bytes: PNG screenshot data
            task: The user's task description
            step: Current step number
            max_steps: Maximum allowed steps
            history: List of previous action descriptions
            
        Returns:
            BrowserAction with the next action to take
            
        Raises:
            Exception: If VLM request fails after retries
        """
        import base64
        
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        prompt = self._build_prompt(task, step, max_steps, history)
        
        # Ensure only this Qwen model is running (VRAM optimization)
        ensure_exclusive_qwen(self.model)
        
        self._emit_status("Thinking...")
        
        for attempt in range(VLM_MAX_RETRIES + 1):
            try:
                return await self._request_action(screenshot_b64, prompt, attempt)
            except (ValidationError, json.JSONDecodeError) as e:
                if attempt < VLM_MAX_RETRIES:
                    self._emit_status(f"Retrying... ({attempt + 1}/{VLM_MAX_RETRIES})")
                    continue
                # Return a wait action on final failure
                return BrowserAction(
                    action="wait",
                    target="2",
                    reasoning=f"Failed to parse response after {VLM_MAX_RETRIES} attempts: {e}"
                )
            except Exception as e:
                raise Exception(f"VLM request failed: {e}")
    
    async def _request_action(
        self,
        screenshot_b64: str,
        prompt: str,
        attempt: int
    ) -> BrowserAction:
        """
        Make the actual request to Ollama.
        
        Args:
            screenshot_b64: Base64-encoded screenshot
            prompt: The system prompt
            attempt: Current attempt number (for logging)
            
        Returns:
            Parsed BrowserAction
        """
        if DEBUG:
            print(f"[DEBUG] Attempt {attempt + 1}: Sending request to {self.ollama_url}")
        
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [screenshot_b64]
                    }
                ],
                "stream": True,
                "format": get_action_schema(),
                "options": {
                    "temperature": 0.2,
                    "num_predict": 2000  # Increased for thinking + JSON
                }
            },
            timeout=VLM_TIMEOUT,
            stream=True
        )
        response.raise_for_status()
        
        # Process streaming response
        full_thinking = ""
        full_content = ""
        
        if DEBUG:
            print(f"{GRAY}[Thinking] ", end="", flush=True)
        
        for line in response.iter_lines():
            if not line:
                continue
            
            chunk = json.loads(line)
            message = chunk.get("message", {})
            
            # Stream thinking tokens
            thinking = message.get("thinking", "")
            if thinking:
                full_thinking += thinking
                self._emit_thinking(thinking)
                if DEBUG:
                    sys.stdout.write(thinking)
                    sys.stdout.flush()
            
            # Collect content
            content = message.get("content", "")
            if content:
                full_content += content
            
            if chunk.get("done"):
                break
        
        if DEBUG:
            print(f"{RESET}")
            if full_content:
                print(f"[DEBUG] Response: {full_content[:200]}...")
            else:
                print(f"[DEBUG] No content, checking thinking for JSON...")
        
        # If content is empty, try to extract JSON from thinking
        if not full_content.strip() and full_thinking:
            import re
            # Look for JSON in thinking
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', full_thinking)
            if json_match:
                full_content = json_match.group()
                if DEBUG:
                    print(f"[DEBUG] Extracted from thinking: {full_content}")
        
        if not full_content.strip():
            raise ValueError("Empty response from VLM")
        
        # Parse with Pydantic - structured output guarantees valid JSON
        action = BrowserAction.model_validate_json(full_content)
        
        if DEBUG:
            print(f"[DEBUG] Parsed action: {action}")
        
        return action
