"""
Browser Controller - Playwright browser lifecycle management.

Handles browser launching, page management, and screenshot capture
with anti-detection measures.
"""

import asyncio
from typing import Optional

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


# Anti-detection configuration
BROWSER_ARGS = [
    '--disable-blink-features=AutomationControlled',
    '--disable-dev-shm-usage',
    '--no-sandbox'
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

VIEWPORT = {"width": 1280, "height": 720}


class BrowserController:
    """
    Manages Playwright browser lifecycle and page operations.
    
    Provides async context manager interface for clean resource management.
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the browser controller.
        
        Args:
            headless: Whether to run browser in headless mode
        """
        self.headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
    
    @property
    def page(self) -> Optional[Page]:
        """Get the current page."""
        return self._page
    
    @property
    def is_running(self) -> bool:
        """Check if browser is running."""
        return self._browser is not None
    
    async def start(self) -> Page:
        """
        Launch browser and create a new page.
        
        Returns:
            The created Page object
        """
        self._playwright = await async_playwright().start()
        
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=BROWSER_ARGS
        )
        
        self._context = await self._browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT,
            locale="en-US"
        )
        
        self._page = await self._context.new_page()
        
        return self._page
    
    async def close(self):
        """Close browser and cleanup resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._context = None
            self._page = None
        
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
    
    async def capture_screenshot(self) -> bytes:
        """
        Capture screenshot of current page.
        
        Returns:
            PNG image data as bytes
        """
        if not self._page:
            return b""
        
        return await self._page.screenshot(type="png")
    
    async def navigate(self, url: str, timeout: int = 30000):
        """
        Navigate to a URL.
        
        Args:
            url: The URL to navigate to
            timeout: Navigation timeout in milliseconds
        """
        if self._page:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    
    async def __aenter__(self) -> "BrowserController":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
