"""
MessageBubble component - Styled chat message bubble for PySide6.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QTextBrowser, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QDesktopServices

import markdown
from pygments.formatters import HtmlFormatter

# Pre-generate CSS for code blocks
CODE_CSS = HtmlFormatter(style='monokai').get_style_defs('.codehilite')

class ResizingTextBrowser(QTextBrowser):
    """A QTextBrowser that automatically resizes to fit its content."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(QDesktopServices.openUrl)
        
        # Style
        self.viewport().setStyleSheet("background: transparent;")
        self.setStyleSheet("background: transparent; border: none;")
        
        self.document().contentsChanged.connect(self.adjust_height)

    def adjust_height(self):
        doc_height = self.document().size().height()
        self.setFixedHeight(int(doc_height) + 10)
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.adjust_height()

class MessageBubble(QFrame):
    """A styled message bubble for User or AI with Markdown support."""
    
    def __init__(self, role: str, text: str = "", is_thinking: bool = False, parent=None):
        super().__init__(parent)
        self.role = role
        self.is_thinking = is_thinking
        self._text = text
        
        self.setObjectName("messageBubble")
        self._setup_ui()
        self._apply_style()
        self.set_text(text) # Render initially
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(0)
        
        # Use Custom Resizing Browser
        self.content_label = ResizingTextBrowser()
        
        if self.is_thinking:
            self.content_label.setFont(QFont("Consolas", 11))
        else:
            self.content_label.setFont(QFont("Segoe UI", 11))
        
        self.content_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.content_label)
        
    def _apply_style(self):
        is_user = self.role == "user"
        
        if self.is_thinking:
            bg_color = "#2a2a2a"
            border_radius = "12px"
            text_color = "#9e9e9e"
        elif is_user:
            bg_color = "#005c4b"
            border_radius = "18px 18px 4px 18px"
            text_color = "#e8eaed"
        else:
            bg_color = "#363636"
            border_radius = "18px 18px 18px 4px"
            text_color = "#e8eaed"
        
        self.setStyleSheet(f"""
            QFrame#messageBubble {{
                background-color: {bg_color};
                border-radius: {border_radius};
            }}
            QTextBrowser {{
                color: {text_color};
            }}
        """)
        
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setMinimumWidth(60)
        self.setMaximumWidth(600) # Slightly wider for code
    
    def set_text(self, text: str, force_markdown: bool = True):
        """Update the message content (for streaming)."""
        self._text = text
        
        # Optimization: Only render Markdown if there are potential markdown characters 
        # or if explicitly forced (like when generation finishes).
        # This avoids expensive regex/html conversion for every plain text token.
        has_markdown = any(c in text for c in ['*', '`', '[', '#', '|', '-', '>'])
        
        if not force_markdown and not has_markdown:
             # Fast path for plain text
             self.content_label.setPlainText(text)
             self.content_label.adjust_height()
             return

        # Convert Markdown to HTML
        html_content = markdown.markdown(
            text, 
            extensions=['fenced_code', 'codehilite', 'nl2br']
        )
        
        styled_html = f"""
        <style>
            body {{ font-family: 'Segoe UI'; font-size: 11pt; margin: 0; padding: 0; }}
            code {{ font-family: 'Consolas', monospace; background-color: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 4px; }}
            pre {{ background-color: #222; padding: 10px; border-radius: 8px; color: #f8f8f2; margin: 5px 0; }}
            {CODE_CSS}
        </style>
        <body>
            {html_content}
        </body>
        """
        
        self.content_label.setHtml(styled_html)
        self.content_label.adjust_height()
    
    def append_text(self, text: str):
        """Append text to the message (for streaming)."""
        self._text += text
        # For streaming, we use the optimized path
        self.set_text(self._text, force_markdown=False)
    
    @property
    def alignment(self):
        """Return the alignment for this bubble."""
        # Note: Code blocks are usually left-aligned, so bubbles with code look better left-aligned even for users?
        # But stick to standard chat UX for now.
        return Qt.AlignRight if self.role == "user" else Qt.AlignLeft
