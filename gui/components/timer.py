from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
)
from PySide6.QtCore import Qt, QTimer

class TimerComponent(QWidget):
    """Flow State Timer Component. Aura Theme."""
    
    def __init__(self):
        super().__init__()
        self.duration = 25 * 60  # 25 minutes default
        self.remaining = self.duration
        self.is_running = False
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_timer)
        self.timer.setInterval(1000)
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Card adds margins
        
        # --- Timer Card ---
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #0f1524;
                border-radius: 12px;
                border: 1px solid #1a2236;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(15)
        
        # Header Row (Title + Edit Button)
        header_layout = QHBoxLayout()
        lbl = QLabel("TIMER")
        lbl.setStyleSheet("color: #e8eaed; font-size: 13px; font-weight: bold; letter-spacing: 1px; background: transparent; border: none;")
        header_layout.addWidget(lbl)
        
        header_layout.addStretch()
        
        self.edit_btn = QPushButton("✎")
        self.edit_btn.setToolTip("Edit Duration")
        self.edit_btn.setCursor(Qt.PointingHandCursor)
        self.edit_btn.setFixedSize(28, 28)
        self.edit_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                color: #8b9bb4;
                border-radius: 14px;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(51, 181, 229, 0.2); 
                color: #33b5e5;
            }
        """)
        self.edit_btn.clicked.connect(self._edit_duration)
        header_layout.addWidget(self.edit_btn)
        
        card_layout.addLayout(header_layout)
        
        # Timer Display
        self.time_display = QLabel("25:00")
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setStyleSheet("""
            color: #33b5e5; 
            font-size: 48px; 
            font-weight: bold;
            font-family: 'Segoe UI', sans-serif;
            background: transparent;
            border: none;
        """)
        card_layout.addWidget(self.time_display)
        
        # Controls (Start/Pause + Reset)
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Start Button
        self.start_btn = QPushButton("START")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: rgba(51, 181, 229, 0.15); 
                color: #33b5e5;
                border: 1px solid #33b5e5;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: rgba(51, 181, 229, 0.3);
                color: white;
            }
            QPushButton:pressed {
                background: #33b5e5;
                color: #05080d;
            }
        """)
        self.start_btn.clicked.connect(self._toggle_timer)
        controls_layout.addWidget(self.start_btn, 1) # Stretch 1
        
        # Reset Button
        self.reset_btn = QPushButton("↺")
        self.reset_btn.setToolTip("Reset Timer")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setFixedSize(40, 40)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                color: #e8eaed;
                border-radius: 20px;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                color: #33b5e5;
            }
        """)
        self.reset_btn.clicked.connect(self._reset_timer)
        controls_layout.addWidget(self.reset_btn)
        
        card_layout.addLayout(controls_layout)
        
        layout.addWidget(card)
        layout.addStretch()

    def _toggle_timer(self):
        if self.is_running:
            self.timer.stop()
            self.start_btn.setText("RESUME")
            self.is_running = False
        else:
            self.timer.start()
            self.start_btn.setText("PAUSE")
            self.is_running = True
            
    def _reset_timer(self):
        self.timer.stop()
        self.is_running = False
        self.start_btn.setText("START")
        self.remaining = self.duration
        self._update_display()
    
    def set_and_start(self, seconds: int, label: str = None):
        """Set timer duration and start it. Called externally (e.g., voice command)."""
        if seconds <= 0:
            return
        self.duration = seconds
        self.remaining = seconds
        self._update_display()
        # Auto-start
        self.timer.start()
        self.start_btn.setText("PAUSE")
        self.is_running = True

    def _edit_duration(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QSpinBox, QLabel, QPushButton
        from PySide6.QtCore import Qt
        
        d = QDialog(self)
        d.setWindowTitle("Set Timer")
        d.setFixedSize(320, 180)
        d.setStyleSheet("""
            QDialog { background-color: #1a2236; color: #e8eaed; }
            QLabel { color: #e8eaed; font-size: 14px; }
            QSpinBox {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                color: #e8eaed;
                font-size: 16px;
                padding: 4px;
            }
            QPushButton {
                background-color: rgba(51, 181, 229, 0.2);
                color: #33b5e5;
                border: 1px solid #33b5e5;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: rgba(51, 181, 229, 0.3); color: white; }
        """)
        
        layout = QVBoxLayout(d)
        
        # Title
        title = QLabel("Set Duration")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Spinners
        spin_layout = QHBoxLayout()
        spin_layout.setSpacing(10)
        
        # Calculate current
        current_h = self.duration // 3600
        current_m = (self.duration % 3600) // 60
        current_s = self.duration % 60
        
        def create_spinner(val, max_val, label_text):
            v_layout = QVBoxLayout()
            s = QSpinBox()
            s.setRange(0, max_val)
            s.setValue(val)
            s.setButtonSymbols(QSpinBox.NoButtons)
            s.setAlignment(Qt.AlignCenter)
            s.setFixedSize(60, 40)
            
            l = QLabel(label_text)
            l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet("font-size: 12px; color: #8b9bb4;")
            
            v_layout.addWidget(s)
            v_layout.addWidget(l)
            spin_layout.addLayout(v_layout)
            return s
            
        h_spin = create_spinner(current_h, 99, "Hrs")
        m_spin = create_spinner(current_m, 59, "Min")
        s_spin = create_spinner(current_s, 59, "Sec")
        
        layout.addLayout(spin_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(d.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(d.reject)
        # Cancel style override
        cancel_btn.setStyleSheet("background: transparent; border: 1px solid #6e6e6e; color: #8b9bb4;")
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        if d.exec():
            h = h_spin.value()
            m = m_spin.value()
            s = s_spin.value()
            total = (h * 3600) + (m * 60) + s
            if total > 0:
                self.duration = total
                self._reset_timer()

    def _update_timer(self):
        if self.remaining > 0:
            self.remaining -= 1
            self._update_display()
        else:
            self.timer.stop()
            self.is_running = False
            self.start_btn.setText("START")
            self.remaining = self.duration
            self._update_display()
            # Could play sound here

    def _update_display(self):
        h = self.remaining // 3600
        m = (self.remaining % 3600) // 60
        s = self.remaining % 60
        
        if h > 0:
            self.time_display.setText(f"{h:02d}:{m:02d}:{s:02d}")
            # Adjust font size if it's too long?
            self.time_display.setStyleSheet("""
                color: #33b5e5; 
                font-size: 36px; 
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
                background: transparent;
                border: none;
            """)
        else:
            self.time_display.setText(f"{m:02d}:{s:02d}")
            self.time_display.setStyleSheet("""
                color: #33b5e5; 
                font-size: 48px; 
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
                background: transparent;
                border: none;
            """)
