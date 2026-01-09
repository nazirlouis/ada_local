"""
ThinkingExpander component - Gemini-style collapsible thinking UI for PySide6.
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, QPushButton, QSizePolicy, QWidget
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QMovie


class ThinkingExpander(QFrame):
    """Gemini-style collapsible thinking UI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("thinkingExpander")
        self._is_expanded = False
        self._animation = None
        self._content_height = 0
        
        self._setup_ui()
        self._apply_style()
        
    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Header (clickable to expand/collapse)
        self.header = QFrame()
        self.header.setObjectName("thinkingHeader")
        self.header.setCursor(Qt.PointingHandCursor)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(8)
        
        # Loading indicator
        self.spinner_label = QLabel("⟳")
        self.spinner_label.setStyleSheet("color: #4F8EF7; font-size: 14px;")
        header_layout.addWidget(self.spinner_label)
        
        # Title
        self.title_label = QLabel("Thinking...")
        self.title_label.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # Expand/Collapse arrow
        self.arrow_label = QLabel("▶")
        self.arrow_label.setStyleSheet("color: #6e6e6e; font-size: 10px;")
        header_layout.addWidget(self.arrow_label)
        
        self.main_layout.addWidget(self.header)
        
        # Content container (collapsible)
        self.content_container = QWidget()
        self.content_container.setMaximumHeight(0)
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(10, 5, 10, 10)
        
        # Thinking log text
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setMinimumHeight(80)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #a0a0a0;
                border: none;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        content_layout.addWidget(self.log_text)
        
        self.main_layout.addWidget(self.content_container)
        
        # Connect click event
        self.header.mousePressEvent = self._on_header_click
        
    def _apply_style(self):
        self.setStyleSheet("""
            QFrame#thinkingExpander {
                background-color: #2a2a2a;
                border-radius: 8px;
                margin-bottom: 8px;
            }
            QFrame#thinkingHeader {
                background-color: transparent;
                border-radius: 8px;
            }
            QFrame#thinkingHeader:hover {
                background-color: #333333;
            }
        """)
        
    def _on_header_click(self, event):
        self.toggle_expanded()
        
    def toggle_expanded(self):
        """Toggle the expanded/collapsed state with animation."""
        self._is_expanded = not self._is_expanded
        
        # Update arrow
        self.arrow_label.setText("▼" if self._is_expanded else "▶")
        
        # Animate content height
        target_height = 220 if self._is_expanded else 0
        
        if self._animation:
            self._animation.stop()
            
        self._animation = QPropertyAnimation(self.content_container, b"maximumHeight", self)
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.setStartValue(self.content_container.maximumHeight())
        self._animation.setEndValue(target_height)
        self._animation.start()
        
    def add_text(self, text: str):
        """Add text to the thinking log."""
        self.log_text.insertPlainText(text)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def complete(self):
        """Mark thinking as complete."""
        self.spinner_label.setText("✓")
        self.spinner_label.setStyleSheet("color: #4CAF50; font-size: 14px;")
        self.title_label.setText("Thinking Complete")
        self.title_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
