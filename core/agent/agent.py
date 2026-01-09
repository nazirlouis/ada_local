"""
Browser Agent - Main orchestrator for AI-powered browser automation.

Composes VLMClient, BrowserController, and ActionExecutor to provide
a complete browser automation agent with the same callback interface
as the original implementation.
"""

import asyncio
import threading
from typing import Callable, Optional

from config import GRAY, RESET
from core.agent.schemas import BrowserAction
from core.agent.vlm_client import VLMClient
from core.agent.browser_controller import BrowserController
from core.agent.action_executor import ActionExecutor


# Configuration
MAX_STEPS = 20
SCREENSHOT_DELAY = 0.5  # seconds between action and screenshot
SCREENSHOT_INTERVAL = 1.0  # seconds between screenshot updates to GUI


class BrowserAgent:
    """
    AI Browser Agent that uses vision-language model to control a web browser.
    
    This is the main orchestrator that composes:
    - VLMClient: Communicates with Ollama VLM
    - BrowserController: Manages Playwright browser
    - ActionExecutor: Executes browser actions
    
    Signals (callbacks):
        on_screenshot: Called with (bytes) when a new screenshot is available
        on_thought: Called with (str) when the AI produces reasoning
        on_action: Called with (BrowserAction) when an action is taken
        on_status: Called with (str) for status updates
        on_complete: Called with (str) when task is complete
        on_error: Called with (str) when an error occurs
    """
    
    def __init__(self):
        """Initialize the browser agent."""
        self._running = False
        self._stop_requested = False
        self._action_history: list[str] = []
        
        # Callbacks (set by caller)
        self.on_screenshot: Optional[Callable[[bytes], None]] = None
        self.on_thought: Optional[Callable[[str], None]] = None
        self.on_action: Optional[Callable[[BrowserAction], None]] = None
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_complete: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    def _emit_status(self, text: str):
        """Emit status update."""
        if self.on_status:
            self.on_status(text)
        print(f"{GRAY}[Agent] {text}{RESET}")
    
    def _emit_thought(self, text: str):
        """Emit thought/reasoning to GUI only (no terminal)."""
        if self.on_thought:
            self.on_thought(text)
    
    def _emit_screenshot(self, data: bytes):
        """Emit screenshot update."""
        if self.on_screenshot:
            self.on_screenshot(data)
    
    def _emit_action(self, action: BrowserAction):
        """Emit action update."""
        if self.on_action:
            self.on_action(action)
        print(f"{GRAY}[Action] {action}{RESET}")
    
    async def _screenshot_loop(self, browser: BrowserController):
        """Background loop to stream screenshots to GUI."""
        while self._running and not self._stop_requested:
            try:
                if browser.page:
                    screenshot = await browser.capture_screenshot()
                    self._emit_screenshot(screenshot)
            except Exception:
                pass
            await asyncio.sleep(SCREENSHOT_INTERVAL)
    
    async def _run_agent(self, task: str):
        """
        Main agent loop.
        
        Args:
            task: The user's task description
        """
        self._running = True
        self._stop_requested = False
        self._action_history = []
        
        try:
            self._emit_status("Launching browser...")
            
            async with BrowserController() as browser:
                # Create VLM client with callbacks
                vlm = VLMClient(
                    on_thinking=self._emit_thought,  # Stream raw thinking tokens
                    on_status=self._emit_status
                )
                
                # Create action executor
                executor = ActionExecutor(browser.page)
                
                # Start screenshot streaming
                screenshot_task = asyncio.create_task(self._screenshot_loop(browser))
                
                self._emit_status("Browser ready. Starting task...")
                # Task is shown in status, not thinking section
                
                # Main agent loop
                for step in range(1, MAX_STEPS + 1):
                    if self._stop_requested:
                        self._emit_status("Stopped by user")
                        break
                    
                    self._emit_status(f"Step {step}/{MAX_STEPS}")
                    
                    # Capture screenshot with delay for page settle
                    await asyncio.sleep(SCREENSHOT_DELAY)
                    screenshot = await browser.capture_screenshot()
                    self._emit_screenshot(screenshot)
                    
                    # Get AI decision
                    action = await vlm.get_action(
                        screenshot_bytes=screenshot,
                        task=task,
                        step=step,
                        max_steps=MAX_STEPS,
                        history=self._action_history
                    )
                    
                    # Step summary goes to status, not thinking (thinking is streaming)
                    self._emit_status(f"Action: {action.action_type.value}")
                    self._emit_action(action)
                    
                    # Track action history to avoid loops
                    self._action_history.append(str(action))
                    
                    # Execute action
                    is_complete, status = await executor.execute(action)
                    self._emit_status(status)
                    
                    if is_complete:
                        if self.on_complete:
                            self.on_complete(action.target)
                        break
                else:
                    # Max steps reached
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
                
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
            self._emit_status(f"Error: {e}")
        finally:
            self._running = False
    
    def run(self, task: str):
        """
        Start the agent in a background thread.
        
        Args:
            task: The user's task description
        """
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
        """Check if agent is currently running."""
        return self._running
