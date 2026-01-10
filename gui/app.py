"""
Main PySide6 application setup and layout using Fluent Widgets.
"""

import threading
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon

from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, FluentIcon as FIF,
    SplashScreen
)

from gui.handlers import ChatHandlers
from core.model_manager import unload_all_models

from gui.styles import AURA_STYLESHEET 

from gui.tabs.dashboard import DashboardView
from gui.tabs.chat import ChatTab
from gui.tabs.planner import PlannerTab
from gui.tabs.settings import SettingsTab
from gui.tabs.briefing import BriefingView
from gui.tabs.browser import BrowserTab
from gui.tabs.home_automation import HomeAutomationTab
from gui.components.system_monitor import SystemMonitor


class LazyTab(QWidget):
    """Placeholder widget that loads the actual tab on demand."""
    def __init__(self, factory, object_name):
        super().__init__()
        self.setObjectName(object_name)
        self.factory = factory
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.actual_widget = None

    def initialize(self):
        if not self.actual_widget:
            self.actual_widget = self.factory()
            self.layout.addWidget(self.actual_widget)
            return self.actual_widget
        return self.actual_widget

class MainWindow(FluentWindow):
    """Main application window using Fluent Design."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A.D.A")
        self.setMinimumSize(1100, 750)
        self.resize(1200, 800)
        
        self.setStyleSheet(AURA_STYLESHEET)
        
        # Initialize handlers
        self.handlers = ChatHandlers(self)
        
        # Add system monitor to title bar
        self._init_system_monitor()
        
        # Initialize sub-interfaces pointers
        self.chat_tab = None
        self.planner_tab = None
        self.briefing_view = None
        self.home_tab = None

        self._init_window()
        self._connect_signals()
        self._init_background()
        
    def _init_window(self):
        # Dashboard is loaded immediately as it's the home screen
        self.dashboard_view = DashboardView()
        self.dashboard_view.setObjectName("dashboardInterface")
        self.dashboard_view.navigate_to.connect(self._navigate_to_tab)
        self.addSubInterface(self.dashboard_view, FIF.LAYOUT, "Dashboard")

        # Lazy load other tabs
        self.chat_lazy = LazyTab(ChatTab, "chatInterface")
        self.planner_lazy = LazyTab(PlannerTab, "plannerInterface")
        # Eager load briefing for startup fetch
        self.briefing_view = BriefingView()
        self.briefing_view.setObjectName("briefingInterface")

        self.home_lazy = LazyTab(HomeAutomationTab, "homeInterface")
        self.browser_lazy = LazyTab(BrowserTab, "browserInterface")
        
        self.addSubInterface(self.chat_lazy, FIF.CHAT, "Chat")
        self.addSubInterface(self.planner_lazy, FIF.CALENDAR, "Planner")
        self.addSubInterface(self.briefing_view, FIF.DATE_TIME, "Briefing")
        self.addSubInterface(self.home_lazy, FIF.HOME, "Home Auto")
        self.addSubInterface(self.browser_lazy, FIF.GLOBE, "Web Agent")
        
        # Settings at bottom
        self.settings_lazy = LazyTab(SettingsTab, "settingsInterface")
        self.addSubInterface(
            self.settings_lazy, FIF.SETTING, "Settings",
            NavigationItemPosition.BOTTOM
        )
        
    def _connect_signals(self):
        """Connect signals. Signals for lazy tabs are connected upon initialization."""
        self.stackedWidget.currentChanged.connect(self._on_tab_changed)

    def _connect_chat_signals(self):
        """Connect ChatTab signals (called when ChatTab is initialized)."""
        if not self.chat_tab:
            return
        self.chat_tab.new_chat_requested.connect(self.handlers.clear_chat)
        self.chat_tab.send_message_requested.connect(self._on_send)
        self.chat_tab.stop_generation_requested.connect(self.handlers.stop_generation)
        self.chat_tab.tts_toggled.connect(self.handlers.toggle_tts)
        self.chat_tab.session_selected.connect(self._on_session_clicked)
        
        self.chat_tab.session_pin_requested.connect(self.handlers.pin_session)
        self.chat_tab.session_rename_requested.connect(self.handlers.rename_session)
        self.chat_tab.session_delete_requested.connect(self.handlers.delete_session)
        
        # Initial sidebar refresh
        self.chat_tab.refresh_sidebar()

    def _on_send(self, text):
        """Forward send request to handlers."""
        self.handlers.send_message(text)
        
    def _on_session_clicked(self, session_id):
        """Load session."""
        self.handlers.load_session(session_id)
    
    def _init_background(self):
        """Initialize app status."""
        self.set_status("Ready")
    
    def _init_system_monitor(self):
        """Add system monitor widget to the title bar, centered with controls on the right."""
        self.system_monitor = SystemMonitor()
        
        # Get the title bar layout
        layout = self.titleBar.hBoxLayout
        
        # dynamic search for min button index to ensure we insert BEFORE the window controls
        min_btn_index = layout.indexOf(self.titleBar.minBtn)
        
        # Insert a stretch to push monitor toward center (after title/icon, before buttons)
        layout.insertStretch(min_btn_index, 1)
        # Insert the system monitor
        layout.insertWidget(min_btn_index + 1, self.system_monitor, 0, Qt.AlignmentFlag.AlignCenter)
        # Insert another stretch after monitor to balance centering
        layout.insertStretch(min_btn_index + 2, 1)
    
    def _on_tab_changed(self, index):
        """Handle lazy loading when switching tabs."""
        widget = self.stackedWidget.widget(index)
        
        if isinstance(widget, LazyTab):
            real_widget = widget.initialize()
            obj_name = widget.objectName()
            
            # Map lazy widget to attribute
            if obj_name == "chatInterface":
                self.chat_tab = real_widget
                self._connect_chat_signals()
            elif obj_name == "plannerInterface":
                self.planner_tab = real_widget
            elif obj_name == "briefingInterface":
                self.briefing_view = real_widget
            elif obj_name == "homeInterface":
                self.home_tab = real_widget
            elif obj_name == "browserInterface":
                # No signals to connect for browser yet
                pass
                
        self.set_status("Ready")
    
    def _navigate_to_tab(self, route_key: str):
        """Navigate to a tab by its object name (route key)."""
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            if widget.objectName() == route_key:
                self.switchTo(widget)
                return
    
    # --- Public Methods for Handlers (Proxy/Facade) ---
    # These now check if the tab exists before calling
    
    def set_status(self, text: str):
        if self.chat_tab: self.chat_tab.set_status(text)
    
    def clear_input(self):
        if self.chat_tab: self.chat_tab.clear_input()
    
    def set_generating_state(self, is_generating: bool):
        if self.chat_tab: self.chat_tab.set_generating_state(is_generating)
    
    def add_message_bubble(self, role: str, text: str, is_thinking: bool = False):
        if self.chat_tab: self.chat_tab.add_message_bubble(role, text, is_thinking)
    
    def add_streaming_widgets(self, thinking_ui, response_bubble):
        if self.chat_tab: self.chat_tab.add_streaming_widgets(thinking_ui, response_bubble)
    
    def clear_chat_display(self):
        if self.chat_tab: self.chat_tab.clear_chat_display()
    
    def refresh_sidebar(self, current_session_id: str = None):
        if self.chat_tab: self.chat_tab.refresh_sidebar(current_session_id)
    
    def scroll_to_bottom(self):
        if self.chat_tab: self.chat_tab.scroll_to_bottom()

    def closeEvent(self, event):
        """Handle application close event."""
        print("[App] Closing application, unloading models...")
        self.set_status("Closing...")
        unload_all_models(sync=True)
        event.accept()


def create_app():
    """Create and return the main window."""
    return MainWindow()
