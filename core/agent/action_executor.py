"""
Action Executor - Execute browser actions via Playwright.

Handles element finding with multiple locator strategies and
action execution with proper error handling.
"""

from typing import Optional
from playwright.async_api import Page, Locator

from core.agent.schemas import BrowserAction, ActionType


class ActionExecutor:
    """
    Executes browser actions using Playwright.
    
    Provides multi-strategy element finding and robust action execution.
    """
    
    def __init__(self, page: Page):
        """
        Initialize the action executor.
        
        Args:
            page: Playwright Page object
        """
        self.page = page
    
    async def find_element(self, description: str, timeout: int = 5000) -> Optional[Locator]:
        """
        Find an element using multiple locator strategies.
        
        Tries various locator methods in order of preference until
        a visible element is found.
        
        Args:
            description: Text description of the element
            timeout: Timeout in milliseconds (not used currently, kept for API compat)
            
        Returns:
            Locator for the found element, or None if not found
        """
        # Strategies in order of preference
        locators = [
            # Text-based
            self.page.get_by_text(description, exact=False),
            # Form elements
            self.page.get_by_placeholder(description, exact=False),
            self.page.get_by_label(description, exact=False),
            # Attributes
            self.page.get_by_title(description, exact=False),
            self.page.get_by_alt_text(description, exact=False),
            # Roles with name
            self.page.get_by_role("button", name=description, exact=False),
            self.page.get_by_role("link", name=description, exact=False),
            self.page.get_by_role("searchbox", name=description, exact=False),
            self.page.get_by_role("textbox", name=description, exact=False),
            self.page.get_by_role("combobox", name=description, exact=False),
        ]
        
        for locator in locators:
            try:
                count = await locator.count()
                if count > 0:
                    element = locator.first
                    if await element.is_visible():
                        return element
            except Exception:
                continue
        
        return None
    
    async def execute(self, action: BrowserAction) -> tuple[bool, str]:
        """
        Execute a browser action.
        
        Args:
            action: The BrowserAction to execute
            
        Returns:
            Tuple of (is_complete, status_message)
            is_complete is True if the task is done
        """
        action_type = action.action_type
        
        try:
            if action_type == ActionType.NAVIGATE:
                return await self._execute_navigate(action)
            
            elif action_type == ActionType.CLICK:
                return await self._execute_click(action)
            
            elif action_type == ActionType.TYPE:
                return await self._execute_type(action)
            
            elif action_type == ActionType.SCROLL:
                return await self._execute_scroll(action)
            
            elif action_type == ActionType.WAIT:
                return await self._execute_wait(action)
            
            elif action_type == ActionType.DONE:
                return True, f"Task complete: {action.target}"
            
            else:
                return False, f"Unknown action type: {action_type}"
                
        except Exception as e:
            return False, f"Action failed: {e}"
    
    async def _execute_navigate(self, action: BrowserAction) -> tuple[bool, str]:
        """Execute navigate action."""
        url = action.target
        # Add https:// if missing
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return False, f"Navigated to {url}"
    
    async def _execute_click(self, action: BrowserAction) -> tuple[bool, str]:
        """Execute click action."""
        element = await self.find_element(action.target)
        
        if element:
            await element.click(timeout=5000)
            return False, f"Clicked: {action.target}"
        else:
            return False, f"Could not find element: {action.target}"
    
    async def _execute_type(self, action: BrowserAction) -> tuple[bool, str]:
        """Execute type action."""
        if action.target and action.value:
            # Strategies:
            # 1. Try fill() - best for inputs, handles clearing
            # 2. Try click() + type() - fallback for non-inputs
            # 3. Just type() - fallback if element not found
            
            element = await self.find_element(action.target)
            
            if element:
                try:
                    # Strategy 1: Attempt to fill
                    await element.fill(action.value)
                    # Often need to press Enter after typing in search bars
                    await self.page.keyboard.press("Enter")
                    return False, f"Filled '{action.value[:20]}...' in {action.target}"
                except Exception:
                    # Strategy 2: Fallback to click + type
                    try:
                        await element.click()
                        await self.page.keyboard.type(action.value)
                        await self.page.keyboard.press("Enter")
                        return False, f"Typed (fallback) '{action.value[:20]}...' in {action.target}"
                    except Exception as e:
                        return False, f"Failed to type in {action.target}: {e}"
            else:
                # Strategy 3: Blind type
                await self.page.keyboard.type(action.value)
                await self.page.keyboard.press("Enter")
                return False, f"Typed (blind): {action.value[:20]}..."
        else:
            # Legacy: target is the text to type
            text = action.target if action.target else (action.value or "")
            await self.page.keyboard.type(text)
            await self.page.keyboard.press("Enter")
            return False, f"Typed: {text[:30]}..."
    
    async def _execute_scroll(self, action: BrowserAction) -> tuple[bool, str]:
        """Execute scroll action."""
        direction = action.target.lower()
        delta = 500 if direction == "down" else -500
        await self.page.mouse.wheel(0, delta)
        return False, f"Scrolled {direction}"
    
    async def _execute_wait(self, action: BrowserAction) -> tuple[bool, str]:
        """Execute wait action."""
        import asyncio
        
        try:
            seconds = float(action.target)
        except (ValueError, TypeError):
            seconds = 1.0
        
        await asyncio.sleep(seconds)
        return False, f"Waited {seconds}s"
