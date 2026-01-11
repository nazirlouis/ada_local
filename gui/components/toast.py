"""
Toast Notification Component - Shows success/failure notifications.
"""

from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont


class ToastNotification(QWidget):
    """A toast notification that slides in from top-right and auto-dismisses."""
    
    def __init__(self, parent, message: str, success: bool = True, duration_ms: int = 3000):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        self._setup_ui(message, success)
        self._setup_animation(duration_ms)
        
    def _setup_ui(self, message: str, success: bool):
        """Setup the toast UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        
        # Icon
        icon = "✓" if success else "✗"
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        
        # Message
        msg_label = QLabel(message)
        msg_label.setFont(QFont("Segoe UI", 11))
        msg_label.setWordWrap(True)
        msg_label.setMaximumWidth(300)
        
        layout.addWidget(icon_label)
        layout.addWidget(msg_label, 1)
        
        # Styling
        bg_color = "#2d5a2d" if success else "#5a2d2d"  # Dark green / dark red
        border_color = "#4CAF50" if success else "#f44336"  # Green / red accent
        text_color = "#ffffff"
        
        self.setStyleSheet(f"""
            ToastNotification {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
            }}
            QLabel {{
                color: {text_color};
                background: transparent;
            }}
        """)
        
        self.adjustSize()
        
    def _setup_animation(self, duration_ms: int):
        """Setup fade-in/out animations."""
        # Opacity effect
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)
        
        # Fade in
        self.fade_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in.setDuration(200)
        self.fade_in.setStartValue(0)
        self.fade_in.setEndValue(1)
        self.fade_in.setEasingCurve(QEasingCurve.OutCubic)
        
        # Fade out
        self.fade_out = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_out.setDuration(300)
        self.fade_out.setStartValue(1)
        self.fade_out.setEndValue(0)
        self.fade_out.setEasingCurve(QEasingCurve.InCubic)
        self.fade_out.finished.connect(self.deleteLater)
        
        # Auto-dismiss timer
        self.dismiss_timer = QTimer(self)
        self.dismiss_timer.setSingleShot(True)
        self.dismiss_timer.timeout.connect(self._start_fade_out)
        self.dismiss_timer.setInterval(duration_ms)
        
    def showEvent(self, event):
        """Position and animate on show."""
        super().showEvent(event)
        
        # Position in top-right of parent
        if self.parent():
            parent_rect = self.parent().rect()
            x = parent_rect.width() - self.width() - 20
            y = 60  # Below title bar
            self.move(x, y)
        
        self.fade_in.start()
        self.dismiss_timer.start()
        
    def _start_fade_out(self):
        """Start fade out animation."""
        self.fade_out.start()
    
    @staticmethod
    def show_toast(parent, message: str, success: bool = True, duration_ms: int = 3000):
        """Static method to show a toast notification."""
        toast = ToastNotification(parent, message, success, duration_ms)
        toast.show()
        return toast
