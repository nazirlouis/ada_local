"""
Agent Tab - Browser automation agent UI component.

Displays:
- Prompt input for user tasks
- Live screenshot of browser (updated every ~1 second)
- AI thoughts/reasoning panel
- Status bar and stop button
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
    QScrollArea, QFrame, QSplitter
)
from PySide6.QtCore import Qt, Signal, Slot, QObject, QByteArray
from PySide6.QtGui import QPixmap, QImage, QFont

from qfluentwidgets import (
    TextEdit, PrimaryPushButton, PushButton, 
    SubtitleLabel, BodyLabel, CardWidget, ScrollArea,
    FluentIcon as FIF, InfoBar, InfoBarPosition
)

from gui.components.thinking_expander import ThinkingExpander


class AgentSignals(QObject):
    """Signals for thread-safe GUI updates from agent."""
    screenshot_ready = Signal(bytes)
    thinking_chunk = Signal(str)  # Streaming thinking tokens
    step_complete = Signal(str)   # Step summary (reasoning)
    status_ready = Signal(str)
    action_ready = Signal(str)    # Action description
    complete = Signal(str)
    error = Signal(str)


class ThoughtItem(QFrame):
    """A single thought/action item in the log."""
    
    def __init__(self, text: str, is_action: bool = False):
        super().__init__()
        self.setObjectName("thoughtItem")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        
        # Icon
        icon_label = QLabel("🤔" if not is_action else "▶️")
        icon_label.setFixedWidth(24)
        layout.addWidget(icon_label)
        
        # Text
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #e0e0e0;" if not is_action else "color: #80cbc4;")
        layout.addWidget(text_label, 1)
        
        self.setStyleSheet("""
            QFrame#thoughtItem {
                background: rgba(30, 30, 46, 0.6);
                border-radius: 8px;
                margin: 2px 0;
            }
        """)


class AgentTab(QWidget):
    """Agent tab with browser automation UI."""
    
    def __init__(self):
        super().__init__()
        self.agent = None
        self.signals = AgentSignals()
        self._current_thinking_expander = None  # Active thinking widget
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Build the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # --- Header ---
        header = SubtitleLabel("🤖 Browser Agent")
        header.setStyleSheet("color: #bb86fc; font-size: 18px; font-weight: bold;")
        main_layout.addWidget(header)
        
        # --- Input Section ---
        input_card = CardWidget()
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(16, 16, 16, 16)
        
        input_label = BodyLabel("Describe the task you want the agent to complete:")
        input_label.setStyleSheet("color: #b0b0b0;")
        input_layout.addWidget(input_label)
        
        self.task_input = TextEdit()
        self.task_input.setPlaceholderText("e.g., Go to google.com and search for 'weather in New York'")
        self.task_input.setFixedHeight(80)
        input_layout.addWidget(self.task_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.run_button = PrimaryPushButton("Run Agent", icon=FIF.PLAY)
        self.run_button.clicked.connect(self._on_run_clicked)
        button_layout.addWidget(self.run_button)
        
        self.stop_button = PushButton("Stop", icon=FIF.CANCEL)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        input_layout.addLayout(button_layout)
        main_layout.addWidget(input_card)
        
        # --- Status Bar ---
        self.status_label = BodyLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #808080;")
        main_layout.addWidget(self.status_label)
        
        # --- Main Content: Screenshot + Thoughts ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Screenshot panel (left)
        screenshot_card = CardWidget()
        screenshot_layout = QVBoxLayout(screenshot_card)
        screenshot_layout.setContentsMargins(8, 8, 8, 8)
        
        screenshot_header = BodyLabel("Browser View")
        screenshot_header.setStyleSheet("color: #bb86fc; font-weight: bold;")
        screenshot_layout.addWidget(screenshot_header)
        
        self.screenshot_label = QLabel()
        self.screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setMinimumSize(640, 360)
        self.screenshot_label.setStyleSheet("""
            QLabel {
                background: #1e1e2e;
                border: 1px solid #333;
                border-radius: 8px;
            }
        """)
        self.screenshot_label.setText("Browser screenshot will appear here")
        self.screenshot_label.setStyleSheet(self.screenshot_label.styleSheet() + "color: #666;")
        screenshot_layout.addWidget(self.screenshot_label, 1)
        
        splitter.addWidget(screenshot_card)
        
        # Thoughts panel (right)
        thoughts_card = CardWidget()
        thoughts_layout = QVBoxLayout(thoughts_card)
        thoughts_layout.setContentsMargins(8, 8, 8, 8)
        
        thoughts_header = BodyLabel("AI Thoughts")
        thoughts_header.setStyleSheet("color: #bb86fc; font-weight: bold;")
        thoughts_layout.addWidget(thoughts_header)
        
        # Scrollable thoughts area
        self.thoughts_scroll = ScrollArea()
        self.thoughts_scroll.setWidgetResizable(True)
        self.thoughts_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)
        
        self.thoughts_container = QWidget()
        self.thoughts_layout = QVBoxLayout(self.thoughts_container)
        self.thoughts_layout.setContentsMargins(4, 4, 4, 4)
        self.thoughts_layout.setSpacing(4)
        self.thoughts_layout.addStretch()
        
        self.thoughts_scroll.setWidget(self.thoughts_container)
        thoughts_layout.addWidget(self.thoughts_scroll, 1)
        
        splitter.addWidget(thoughts_card)
        
        # Set initial sizes (60% screenshot, 40% thoughts)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter, 1)
    
    def _connect_signals(self):
        """Connect agent signals to UI update slots."""
        self.signals.screenshot_ready.connect(self._update_screenshot)
        self.signals.thinking_chunk.connect(self._on_thinking_chunk)
        self.signals.step_complete.connect(self._on_step_complete)
        self.signals.status_ready.connect(self._update_status)
        self.signals.action_ready.connect(self._add_action)
        self.signals.complete.connect(self._on_complete)
        self.signals.error.connect(self._on_error)
    
    def _on_run_clicked(self):
        """Start the agent with the given task."""
        task = self.task_input.toPlainText().strip()
        if not task:
            InfoBar.warning(
                title="No task",
                content="Please enter a task description",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        # Clear previous thoughts
        self._clear_thoughts()
        
        # Import and create agent
        from core.agent import BrowserAgent
        
        self.agent = BrowserAgent()
        
        # Connect callbacks (using signals for thread safety)
        self.agent.on_screenshot = lambda data: self.signals.screenshot_ready.emit(data)
        self.agent.on_thought = lambda text: self.signals.thinking_chunk.emit(text)
        self.agent.on_status = lambda text: self.signals.status_ready.emit(text)
        self.agent.on_action = lambda action: self._handle_action(action)
        self.agent.on_complete = lambda result: self.signals.complete.emit(result)
        self.agent.on_error = lambda err: self.signals.error.emit(err)
        
        # Update UI state
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.task_input.setEnabled(False)
        
        # Run agent
        self.agent.run(task)
    
    def _on_stop_clicked(self):
        """Stop the running agent."""
        if self.agent:
            self.agent.stop()
    
    def _clear_thoughts(self):
        """Clear all thought items."""
        self._current_thinking_expander = None
        while self.thoughts_layout.count() > 1:  # Keep the stretch
            item = self.thoughts_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _ensure_thinking_expander(self):
        """Ensure we have a thinking expander for the current step."""
        if not self._current_thinking_expander:
            self._current_thinking_expander = ThinkingExpander()
            self._current_thinking_expander.setVisible(True)
            # Insert before the stretch
            self.thoughts_layout.insertWidget(self.thoughts_layout.count() - 1, self._current_thinking_expander)
        return self._current_thinking_expander
    
    def _handle_action(self, action):
        """Handle action from agent - emits both step_complete and action_ready."""
        # Complete the current thinking expander and emit step summary
        if self._current_thinking_expander:
            self._current_thinking_expander.complete()
            self._current_thinking_expander = None
        
        # Emit action
        self.signals.action_ready.emit(f"{action.action_type.value}: {action.target}")
    
    @Slot(bytes)
    def _update_screenshot(self, data: bytes):
        """Update the screenshot display."""
        image = QImage.fromData(QByteArray(data))
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            # Scale to fit while maintaining aspect ratio
            scaled = pixmap.scaled(
                self.screenshot_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.screenshot_label.setPixmap(scaled)
    
    @Slot(str)
    def _on_thinking_chunk(self, text: str):
        """Stream thinking tokens to the ThinkingExpander."""
        expander = self._ensure_thinking_expander()
        expander.add_text(text)
        # Scroll to bottom
        self.thoughts_scroll.verticalScrollBar().setValue(
            self.thoughts_scroll.verticalScrollBar().maximum()
        )
    
    @Slot(str)
    def _on_step_complete(self, text: str):
        """Handle step completion - finalize current thinking."""
        if self._current_thinking_expander:
            self._current_thinking_expander.complete()
            self._current_thinking_expander = None
    
    @Slot(str)
    def _add_action(self, text: str):
        """Add an action to the log."""
        item = ThoughtItem(text, is_action=True)
        self.thoughts_layout.insertWidget(self.thoughts_layout.count() - 1, item)
        self.thoughts_scroll.verticalScrollBar().setValue(
            self.thoughts_scroll.verticalScrollBar().maximum()
        )
    
    @Slot(str)
    def _update_status(self, text: str):
        """Update status label."""
        self.status_label.setText(f"Status: {text}")
    
    @Slot(str)
    def _on_complete(self, result: str):
        """Handle agent completion."""
        self._reset_ui_state()
        InfoBar.success(
            title="Complete",
            content=result[:100] if result else "Task completed",
            parent=self,
            position=InfoBarPosition.TOP
        )
    
    @Slot(str)
    def _on_error(self, error: str):
        """Handle agent error."""
        self._reset_ui_state()
        InfoBar.error(
            title="Error",
            content=error[:100],
            parent=self,
            position=InfoBarPosition.TOP
        )
    
    def _reset_ui_state(self):
        """Reset UI to idle state."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.task_input.setEnabled(True)
        self.status_label.setText("Status: Idle")
