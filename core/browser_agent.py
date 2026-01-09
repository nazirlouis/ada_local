"""
Browser Agent - AI-powered browser automation using Qwen3-VL:2B vision model.

This module provides a BrowserAgent class that:
1. Launches a headless Playwright browser
2. Captures screenshots and sends them to qwen3-vl:2b via Ollama
3. Parses the model's response for actions (click, type, scroll, navigate, done)
4. Executes actions via Playwright
5. Streams screenshots and thoughts to the GUI
"""

import asyncio
import base64
import json
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import requests
from playwright.async_api import async_playwright, Page, Browser

from config import OLLAMA_URL, GRAY, RESET
from core.model_manager import ensure_exclusive_qwen


# --- Configuration ---
AGENT_VLM_MODEL = "qwen3-vl:4b"
AGENT_MAX_STEPS = 20
AGENT_SCREENSHOT_DELAY = 0.5  # seconds between action and screenshot
SCREENSHOT_INTERVAL = 1.0  # seconds between screenshot updates to GUI
DEBUG_VLM = True  # Enable to see raw model responses


class ActionType(Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    DONE = "done"
    WAIT = "wait"


@dataclass
class AgentAction:
    """Represents an action the agent wants to perform."""
    action_type: ActionType
    target: str = ""  # Element description for click, text for type, url for navigate
    value: str = ""   # Additional value (e.g., direction for scroll)
    reasoning: str = ""  # AI's explanation for this action


class BrowserAgent:
    """
    AI Browser Agent that uses vision-language model to control a web browser.
    
    Signals (callbacks):
        on_screenshot: Called with (bytes) when a new screenshot is available
        on_thought: Called with (str) when the AI produces reasoning
        on_action: Called with (AgentAction) when an action is taken
        on_status: Called with (str) for status updates
        on_complete: Called with (str) when task is complete
        on_error: Called with (str) when an error occurs
    """
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._running = False
        self._stop_requested = False
        self._screenshot_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_screenshot: Optional[Callable[[bytes], None]] = None
        self.on_thought: Optional[Callable[[str], None]] = None
        self.on_action: Optional[Callable[[AgentAction], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_complete: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
    def _emit_status(self, text: str):
        if self.on_status:
            self.on_status(text)
        print(f"{GRAY}[Agent] {text}{RESET}")
    
    def _emit_thought(self, text: str):
        if self.on_thought:
            self.on_thought(text)
        print(f"{GRAY}[Thought] {text}{RESET}")
    
    def _emit_screenshot(self, data: bytes):
        if self.on_screenshot:
            self.on_screenshot(data)
    
    def _emit_action(self, action: AgentAction):
        if self.on_action:
            self.on_action(action)
        print(f"{GRAY}[Action] {action.action_type.value}: {action.target}{RESET}")
    
    async def _capture_screenshot(self) -> bytes:
        """Capture current page screenshot."""
        if self.page:
            return await self.page.screenshot(type="png")
        return b""
    
    async def _send_to_vlm(self, screenshot_bytes: bytes, task: str, step: int) -> str:
        """Send screenshot to qwen3-vl:2b and get response."""
        # Encode screenshot as base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        
        # Build the prompt
        prompt = f"""You are a browser automation agent. Your task is: {task}

Current step: {step}/{AGENT_MAX_STEPS}

Look at this screenshot of a web browser and decide what action to take next.

You must respond with EXACTLY ONE action in this JSON format:
{{"action": "ACTION_TYPE", "target": "description or value", "reasoning": "why you chose this"}}

Available actions:
- {{"action": "navigate", "target": "https://example.com", "reasoning": "..."}} - Go to a URL
- {{"action": "click", "target": "description of element to click", "reasoning": "..."}} - Click an element
- {{"action": "type", "target": "text to type", "reasoning": "..."}} - Type text (assumes a text field is focused)
- {{"action": "scroll", "target": "down" or "up", "reasoning": "..."}} - Scroll the page
- {{"action": "wait", "target": "1", "reasoning": "..."}} - Wait for seconds
- {{"action": "done", "target": "result summary", "reasoning": "..."}} - Task is complete

IMPORTANT: 
- For click actions, describe the element visually (e.g., "the blue Search button", "the login link in the top right")
- Be specific about what you see and why you're taking the action
- If the task is complete, use "done" action

Respond with ONLY the JSON, no other text."""

        try:
            if DEBUG_VLM:
                print(f"[DEBUG] Sending request to {OLLAMA_URL}/chat with model {AGENT_VLM_MODEL}")
                print(f"[DEBUG] Screenshot size: {len(screenshot_b64)} bytes (base64)")
            
            # Ensure only this Qwen model is running
            ensure_exclusive_qwen(AGENT_VLM_MODEL)
            
            response = requests.post(
                f"{OLLAMA_URL}/chat",
                json={
                    "model": AGENT_VLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [screenshot_b64]
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                },
                timeout=120
            )
            
            if DEBUG_VLM:
                print(f"[DEBUG] Response status: {response.status_code}")
                print(f"[DEBUG] Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            result = response.json()
            
            if DEBUG_VLM:
                print(f"[DEBUG] Full API response: {result}")
            
            content = result.get("message", {}).get("content", "")
            if not content:
                print(f"[DEBUG] WARNING: Empty content in response!")
                print(f"[DEBUG] Full result structure: {result.keys()}")
            
            return content
        except Exception as e:
            print(f"[DEBUG] VLM request exception: {type(e).__name__}: {e}")
            raise Exception(f"VLM request failed: {e}")
    
    def _parse_action(self, response: str) -> AgentAction:
        """Parse the VLM response into an AgentAction."""
        # Debug: print raw response
        if DEBUG_VLM:
            print(f"\n{'='*60}")
            print(f"[DEBUG] Raw VLM Response:")
            print(f"{'='*60}")
            print(response)
            print(f"{'='*60}\n")
        
        # Try to extract JSON from response
        # Handle thinking model output (may have <think> tags)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        if DEBUG_VLM and cleaned != response:
            print(f"[DEBUG] After removing <think> tags:")
            print(cleaned[:500] if len(cleaned) > 500 else cleaned)
            print()
        
        # Find JSON in response - try pattern with "action" key first
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', cleaned)
        if not json_match:
            # Try finding any JSON object
            json_match = re.search(r'\{[^{}]*\}', cleaned)
        
        if not json_match:
            print(f"[DEBUG] No JSON found in response!")
            return AgentAction(
                action_type=ActionType.WAIT,
                target="1",
                reasoning="Could not parse response, waiting..."
            )
        
        try:
            json_str = json_match.group()
            if DEBUG_VLM:
                print(f"[DEBUG] Extracted JSON: {json_str}")
            
            data = json.loads(json_str)
            action_str = data.get("action", "wait").lower()
            target = data.get("target", "")
            reasoning = data.get("reasoning", "")
            
            if DEBUG_VLM:
                print(f"[DEBUG] Parsed action: {action_str}, target: {target}")
            
            action_type = ActionType.WAIT
            if action_str == "navigate":
                action_type = ActionType.NAVIGATE
            elif action_str == "click":
                action_type = ActionType.CLICK
            elif action_str == "type":
                action_type = ActionType.TYPE
            elif action_str == "scroll":
                action_type = ActionType.SCROLL
            elif action_str == "done":
                action_type = ActionType.DONE
            elif action_str == "wait":
                action_type = ActionType.WAIT
            
            return AgentAction(
                action_type=action_type,
                target=str(target),
                reasoning=reasoning
            )
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {e}")
            return AgentAction(
                action_type=ActionType.WAIT,
                target="1",
                reasoning="JSON parse error, waiting..."
            )
    
    async def _execute_action(self, action: AgentAction) -> bool:
        """Execute an action. Returns True if task is complete."""
        if not self.page:
            return True
        
        try:
            if action.action_type == ActionType.NAVIGATE:
                self._emit_status(f"Navigating to {action.target}")
                await self.page.goto(action.target, wait_until="domcontentloaded", timeout=30000)
                
            elif action.action_type == ActionType.CLICK:
                self._emit_status(f"Clicking: {action.target}")
                # Use AI-described element - try to find by text or role
                # First try exact text match
                try:
                    await self.page.get_by_text(action.target, exact=False).first.click(timeout=5000)
                except:
                    # Try by role with name
                    try:
                        await self.page.get_by_role("button", name=action.target).first.click(timeout=3000)
                    except:
                        try:
                            await self.page.get_by_role("link", name=action.target).first.click(timeout=3000)
                        except:
                            # Last resort: click at center of visible element matching description
                            self._emit_thought(f"Could not find element: {action.target}")
                
            elif action.action_type == ActionType.TYPE:
                self._emit_status(f"Typing: {action.target[:30]}...")
                await self.page.keyboard.type(action.target)
                await self.page.keyboard.press("Enter")
                
            elif action.action_type == ActionType.SCROLL:
                direction = action.target.lower()
                self._emit_status(f"Scrolling {direction}")
                if direction == "down":
                    await self.page.mouse.wheel(0, 500)
                else:
                    await self.page.mouse.wheel(0, -500)
                    
            elif action.action_type == ActionType.WAIT:
                wait_time = float(action.target) if action.target else 1
                self._emit_status(f"Waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                
            elif action.action_type == ActionType.DONE:
                self._emit_status("Task complete!")
                return True
                
        except Exception as e:
            self._emit_thought(f"Action failed: {e}")
        
        return False
    
    async def _screenshot_loop(self):
        """Background loop to stream screenshots to GUI."""
        while self._running and not self._stop_requested:
            try:
                if self.page:
                    screenshot = await self._capture_screenshot()
                    self._emit_screenshot(screenshot)
            except:
                pass
            await asyncio.sleep(SCREENSHOT_INTERVAL)
    
    async def _run_agent(self, task: str):
        """Main agent loop."""
        self._running = True
        self._stop_requested = False
        
        try:
            self._emit_status("Launching browser...")
            
            async with async_playwright() as p:
                self.browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox'
                    ]
                )
                context = await self.browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    locale="en-US"
                )
                self.page = await context.new_page()
                
                # Start screenshot streaming
                screenshot_task = asyncio.create_task(self._screenshot_loop())
                
                self._emit_status("Browser ready. Starting task...")
                self._emit_thought(f"Task: {task}")
                
                for step in range(1, AGENT_MAX_STEPS + 1):
                    if self._stop_requested:
                        self._emit_status("Stopped by user")
                        break
                    
                    self._emit_status(f"Step {step}/{AGENT_MAX_STEPS}")
                    
                    # Capture screenshot
                    await asyncio.sleep(AGENT_SCREENSHOT_DELAY)
                    screenshot = await self._capture_screenshot()
                    self._emit_screenshot(screenshot)
                    
                    # Get AI decision
                    self._emit_status("Thinking...")
                    try:
                        response = await self._send_to_vlm(screenshot, task, step)
                        action = self._parse_action(response)
                        
                        self._emit_thought(f"Step {step}: {action.reasoning}")
                        self._emit_action(action)
                        
                        # Execute action
                        is_complete = await self._execute_action(action)
                        
                        if is_complete:
                            if self.on_complete:
                                self.on_complete(action.target)
                            break
                            
                    except Exception as e:
                        self._emit_thought(f"Error: {e}")
                        await asyncio.sleep(2)
                
                else:
                    self._emit_status("Max steps reached")
                    if self.on_complete:
                        self.on_complete("Task incomplete - max steps reached")
                
                # Stop screenshot streaming
                self._running = False
                screenshot_task.cancel()
                try:
                    await screenshot_task
                except asyncio.CancelledError:
                    pass
                
                await self.browser.close()
                
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
            self._emit_status(f"Error: {e}")
        finally:
            self._running = False
            self.browser = None
            self.page = None
    
    def run(self, task: str):
        """Start the agent in a background thread."""
        def run_async():
            asyncio.run(self._run_agent(task))
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
    
    def stop(self):
        """Request the agent to stop."""
        self._stop_requested = True
        self._emit_status("Stop requested...")
    
    @property
    def is_running(self) -> bool:
        return self._running
