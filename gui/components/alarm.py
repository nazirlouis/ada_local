from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QTimer, QTime
from qfluentwidgets import MessageBoxBase, SubtitleLabel
from qfluentwidgets.components.date_time.time_picker import TimePicker
from core.tasks import task_manager
import datetime

class AddAlarmDialog(MessageBoxBase):
    """Custom Dialog for adding alarms."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel("New Alarm", self)
        self.viewLayout.addWidget(self.titleLabel)
        
        # Custom Time Input (HH:MM AM/PM)
        time_layout = QHBoxLayout()
        time_layout.setSpacing(5)
        
        # Style for spinboxes
        self.spin_style = """
            QSpinBox {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 5px;
                color: #e8eaed;
                font-size: 16px;
                padding: 5px;
            }
            QSpinBox:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """
        
        from PySide6.QtWidgets import QSpinBox, QComboBox
        
        self.hour_spin = QSpinBox()
        self.hour_spin.setRange(1, 12)
        self.hour_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.hour_spin.setAlignment(Qt.AlignCenter)
        self.hour_spin.setFixedSize(60, 40)
        self.hour_spin.setStyleSheet(self.spin_style)
        
        self.minute_spin = QSpinBox()
        self.minute_spin.setRange(0, 59)
        self.minute_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.minute_spin.setAlignment(Qt.AlignCenter)
        self.minute_spin.setFixedSize(60, 40)
        self.minute_spin.setStyleSheet(self.spin_style)
        # Pad with 0
        self.minute_spin.setPrefix("") 
        # Getting leading zeros in QSpinBox is tricky without subclass or delegate, 
        # but we can just let it be int for now or set suffix.
        # Actually QSpinBox doesn't format text by default to 00. 
        # Let's simple use it as is, maybe find a way to pad later if needed.
        
        self.ampm_combo = QComboBox()
        self.ampm_combo.addItems(["AM", "PM"])
        self.ampm_combo.setFixedSize(70, 40)
        self.ampm_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 5px;
                color: #e8eaed;
                font-size: 16px;
                padding-left: 10px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        
        colon = QLabel(":")
        colon.setStyleSheet("color: #e8eaed; font-size: 20px; font-weight: bold;")
        
        time_layout.addStretch()
        time_layout.addWidget(self.hour_spin)
        time_layout.addWidget(colon)
        time_layout.addWidget(self.minute_spin)
        time_layout.addWidget(self.ampm_combo)
        time_layout.addStretch()
        
        self.viewLayout.addLayout(time_layout)
        
        # Set current time
        now = QTime.currentTime()
        h = now.hour()
        m = now.minute()
        am = True
        
        if h >= 12:
            am = False
            if h > 12:
                h -= 12
        elif h == 0:
            h = 12
            
        self.hour_spin.setValue(h)
        self.minute_spin.setValue(m)
        self.ampm_combo.setCurrentIndex(0 if am else 1)
        
        self.yesButton.setText("Set Alarm")
        self.cancelButton.setText("Cancel")

    def get_time(self):
        h = self.hour_spin.value()
        m = self.minute_spin.value()
        is_pm = self.ampm_combo.currentText() == "PM"
        
        if is_pm and h != 12:
            h += 12
        elif not is_pm and h == 12:
            h = 0
            
        return QTime(h, m)

class AlarmComponent(QWidget):
    """Alarm Component for setting reminders. Aura Theme."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._load_alarms()
        
        self.check_timer = QTimer(self)
        self.check_timer.timeout.connect(self._check_alarms)
        self.check_timer.start(5000) 

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Card
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #0f1524;
                border-radius: 12px;
                border: 1px solid #1a2236;
            }
        """)
        self.card_layout = QVBoxLayout(card)
        self.card_layout.setContentsMargins(20, 20, 20, 20)
        self.card_layout.setSpacing(15)
        
        # Header
        header = QHBoxLayout()
        lbl = QLabel("ALARMS")
        lbl.setStyleSheet("color: #e8eaed; font-size: 13px; font-weight: bold; letter-spacing: 1px; background: transparent; border: none;")
        header.addWidget(lbl)
        header.addStretch()
        
        add_btn = QPushButton("+")
        add_btn.setFixedSize(28, 28)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(51, 181, 229, 0.1);
                color: #33b5e5;
                border: 1px solid #33b5e5;
                border-radius: 14px;
                font-weight: bold;
                font-size: 18px;
                padding-bottom: 2px;
            }
            QPushButton:hover {
                background-color: rgba(51, 181, 229, 0.3);
                color: white;
            }
        """)
        add_btn.clicked.connect(self._add_alarm_dialog)
        header.addWidget(add_btn)
        
        self.card_layout.addLayout(header)
        
        # List
        self.alarm_list = QListWidget()
        self.alarm_list.setStyleSheet("background: transparent; border: none; outline: none;")
        self.card_layout.addWidget(self.alarm_list)
        
        layout.addWidget(card)
        
    def _add_alarm_dialog(self):
        w = AddAlarmDialog(self.window())
        if w.exec():
            qtime = w.get_time()
            time_str = qtime.toString("HH:mm")
            task_manager.add_alarm(time_str, "Alarm")
            self._load_alarms()

    def _load_alarms(self):
        self.alarm_list.clear()
        alarms = task_manager.get_alarms()
        for a in alarms:
            self._create_alarm_item(a)
    
    def reload(self):
        """Reload alarms from database. Called externally after voice command."""
        self._load_alarms()

    def _create_alarm_item(self, alarm):
        item = QListWidgetItem()
        from PySide6.QtCore import QSize
        item.setSizeHint(QSize(0, 45))
        
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Time Label
        time_24 = alarm['time']
        try:
            display_time = datetime.datetime.strptime(time_24, "%H:%M").strftime("%I:%M %p").lstrip("0")
        except:
            display_time = time_24
            
        lbl = QLabel(display_time)
        lbl.setStyleSheet("color: #e8eaed; font-size: 16px; font-weight: 500;")
        layout.addWidget(lbl)
        
        layout.addStretch()
        
        # Delete Btn
        del_btn = QPushButton("×")
        del_btn.setFixedSize(24, 24)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setStyleSheet("""
            QPushButton { color: #6e6e6e; background: transparent; border: none; font-size: 18px; font-weight: bold; }
            QPushButton:hover { color: #ef5350; }
        """)
        a_id = alarm['id']
        del_btn.clicked.connect(lambda checked=False, aid=a_id: self._delete_alarm(aid))
        layout.addWidget(del_btn)
        
        self.alarm_list.addItem(item)
        self.alarm_list.setItemWidget(item, widget)

    def _delete_alarm(self, alarm_id):
        task_manager.delete_alarm(alarm_id)
        self._load_alarms()

    def _check_alarms(self):
        now = datetime.datetime.now().strftime("%H:%M")
        alarms = task_manager.get_alarms()
        
        for a in alarms:
            if a['time'] == now:
                # Play sound
                pass 
