import asyncio
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QScrollArea, QGridLayout, QPushButton
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor
from qfluentwidgets import (
    CardWidget, TitleLabel, BodyLabel, 
    FluentIcon as FIF, IconWidget, SwitchButton, Slider,
    ColorPickerButton, ToolButton
)

from core.kasa_control import kasa_manager

class DataFetchThread(QThread):
    devices_found = Signal(list)
    
    def run(self):
        # Run async discovery in a synchronous wrapper
        try:
            print("[HomeAutomation] Starting Kasa device discovery...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            devices = loop.run_until_complete(kasa_manager.discover_devices())
            loop.close()
            print(f"[HomeAutomation] Found {len(devices)} devices")
            self.devices_found.emit(devices)
        except Exception as e:
            print(f"[HomeAutomation] Discovery error: {e}")
            self.devices_found.emit([])

class ActionThread(QThread):
    finished = Signal(bool)
    
    def __init__(self, action, *args):
        super().__init__()
        self.action = action
        self.args = args
        
    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = False
            if self.action == "on":
                success = loop.run_until_complete(kasa_manager.turn_on(*self.args))
            elif self.action == "off":
                success = loop.run_until_complete(kasa_manager.turn_off(*self.args))
            elif self.action == "brightness":
                success = loop.run_until_complete(kasa_manager.set_brightness(*self.args))
            elif self.action == "color":
                success = loop.run_until_complete(kasa_manager.set_hsv(*self.args))
            loop.close()
            self.finished.emit(success)
        except Exception as e:
            print(f"[HomeAutomation] Action '{self.action}' error: {e}")
            self.finished.emit(False)

class DeviceCard(QFrame):
    """
    Card representing a single smart device.
    """
    def __init__(self, device_info, parent=None):
        super().__init__(parent)
        self.device_info = device_info
        self.ip = device_info['ip']
        self.is_bulb = "Bulb" in device_info.get("type", "") or device_info.get("brightness") is not None
        
        self.setFixedSize(300, 160)
        self.setStyleSheet("""
            DeviceCard {
                background-color: #1a2236;
                border: 1px solid #2a3556;
                border-radius: 20px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header: Icon + Toggle
        header = QHBoxLayout()
        
        # Icon Box
        icon_box = QFrame()
        icon_box.setFixedSize(40, 40)
        icon_box.setStyleSheet("background-color: #232d45; border-radius: 12px;")
        ib_layout = QVBoxLayout(icon_box)
        ib_layout.setAlignment(Qt.AlignCenter)
        ib_layout.setContentsMargins(0,0,0,0)
        
        icon = FIF.BRIGHTNESS if self.is_bulb else FIF.TILES
        iw = IconWidget(icon)
        iw.setFixedSize(20, 20)
        ib_layout.addWidget(iw)
        
        header.addWidget(icon_box)
        header.addStretch()
        
        self.toggle = SwitchButton()
        self.toggle.setChecked(device_info['is_on'])
        self.toggle.checkedChanged.connect(self._on_toggle)
        header.addWidget(self.toggle)
        
        layout.addLayout(header)
        
        # Info
        name_label = QLabel(device_info['alias'])
        name_label.setStyleSheet("color: white; font-weight: bold; font-size: 16px; background: transparent;")
        
        status_label = QLabel("ONLINE")
        status_label.setStyleSheet("color: #6e7a8e; font-size: 11px; font-weight: bold; spacing: 2px; background: transparent;")
        
        layout.addWidget(name_label)
        layout.addWidget(status_label)
        
        layout.addWidget(name_label)
        layout.addWidget(status_label)
        
        # Color & Brightness Controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(0, 5, 0, 0)
        
        # Brightness Bar (Only for bulbs)
        if self.is_bulb:
            self.slider = Slider(Qt.Horizontal)
            self.slider.setRange(0, 100)
            val = device_info.get('brightness')
            self.slider.setValue(val if val is not None else 100)
            self.slider.sliderReleased.connect(self._on_brightness_change)
            ctrl_layout.addWidget(self.slider)
        
        # Color Picker (Only for color bulbs)
        if device_info.get('is_color'):
            self.color_btn = ColorPickerButton(QColor("#ffffff"), "Color")
            self.color_btn.setFixedSize(30, 24)
            # Use QColorDialog logic via signal usually, or custom
            self.color_btn.colorChanged.connect(self._on_color_changed)
            ctrl_layout.addWidget(self.color_btn)
            
        if self.is_bulb:
            layout.addLayout(ctrl_layout)
        else:
            layout.addStretch()
            
    def _on_toggle(self, checked):
        action = "on" if checked else "off"
        self.worker = ActionThread(action, self.ip)
        self.worker.start()
        
    def _on_brightness_change(self):
        val = self.slider.value()
        self.worker_b = ActionThread("brightness", self.ip, val)
        self.worker_b.start()

    def _on_color_changed(self, color):
        # Convert QColor to HSV
        h = color.hsvHue()
        s = int(color.hsvSaturationF() * 100)
        v = int(color.valueF() * 100)
        
        # Kasa expects h(0-360), s(0-100), v(0-100)
        self.worker_c = ActionThread("color", self.ip, h, s, v)
        self.worker_c.start()

class HomeAutomationTab(QWidget):
    """
    Environmental Control Dashboard.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("homeAutomationView")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        # Header Section
        self._setup_header(main_layout)
        
        # Tabs / Filter
        self._setup_filters(main_layout)
        
        # Device Grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        
        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.scroll.setWidget(self.grid_widget)
        main_layout.addWidget(self.scroll)
        
        # Load Devices
        self._load_devices()

    def _setup_header(self, parent_layout):
        header = QHBoxLayout()
        
        text_layout = QVBoxLayout()
        title = TitleLabel("Environmental Control", self)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        
        sub = BodyLabel("Localized automation interface.", self)
        sub.setStyleSheet("color: #6e7a8e; font-size: 14px;")
        
        text_layout.addWidget(title)
        text_layout.addWidget(sub)
        
        header.addLayout(text_layout)
        header.addStretch()
        
        # Refresh Button
        refresh_btn = ToolButton(FIF.SYNC, self)
        refresh_btn.setToolTip("Refresh Devices")
        refresh_btn.clicked.connect(self._load_devices)
        header.addWidget(refresh_btn)
        
        header.addSpacing(10)

        # HA Bubble
        ha_bubble = QLabel("•  PERIMETER SECURE") # Closest match to image, or custom text
        # User requested: "Home Assistant coming soon!"
        ha_bubble.setText("⬤  Home Assistant Coming Soon!")
        ha_bubble.setStyleSheet("""
            background-color: #0d121d; 
            color: #33b5e5; 
            border: 1px solid #1a2236; 
            border-radius: 18px; 
            padding: 8px 20px; 
            font-weight: bold;
            font-size: 12px;
        """)
        header.addWidget(ha_bubble)
        
        parent_layout.addLayout(header)

    def _setup_filters(self, parent_layout):
        self.filter_layout = QHBoxLayout()
        self.filter_layout.setSpacing(15)
        self.filter_layout.addStretch()
        parent_layout.addLayout(self.filter_layout)

    def _update_filters(self):
        # Clear existing filters (except stretch)
        while self.filter_layout.count() > 1:
            child = self.filter_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add "All" + Rooms
        rooms = ["All"] + sorted(list(self.room_groups.keys()))
        
        # Insert before stretch
        for i, room in enumerate(rooms):
            btn = QPushButton(room)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, r=room: self._filter_grid(r))
            
            # Style based on active state (handled in _filter_grid primarily, but set initial)
            if i == 0:
                btn.setChecked(True)
                self.current_filter = room
            
            # Custom Style
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #1a2236; 
                    color: #6e7a8e; 
                    border-radius: 15px; 
                    padding: 8px 20px;
                    border: none;
                    font-weight: bold;
                }}
                QPushButton:checked {{
                    background-color: #33b5e5; 
                    color: #0f1524; 
                }}
                QPushButton:hover {{
                    background-color: #232d45;
                }}
            """)
            self.filter_layout.insertWidget(i, btn)

    def _filter_grid(self, room_name):
        # Update button styles logic:
        # Iterate buttons, set checked only if text matches
        for i in range(self.filter_layout.count() - 1): # exclude stretch
            btn = self.filter_layout.itemAt(i).widget()
            if isinstance(btn, QPushButton): # Safety check
                if btn.text() == room_name:
                    btn.setChecked(True)
                else:
                    btn.setChecked(False)

        # Clear Grid
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)

        # Get devices
        if room_name == "All":
            devices = self.all_devices
        else:
            devices = self.room_groups.get(room_name, [])

        if not devices:
            return

        row = 0
        col = 0
        max_cols = 3 
        
        for dev in devices:
            card = DeviceCard(dev)
            self.grid_layout.addWidget(card, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _load_devices(self):
        # Skip if already loading
        if hasattr(self, 'loader') and self.loader and self.loader.isRunning():
            print("[HomeAutomation] Skipping - discovery already in progress")
            return
            
        self.loader = DataFetchThread()
        self.loader.devices_found.connect(self._on_devices_loaded)
        self.loader.finished.connect(self.loader.deleteLater)
        self.loader.start()
        
    def _on_devices_loaded(self, devices):
        self.all_devices = devices
        self.room_groups = {}
        
        # Categorize
        keywords = {
            "Office": ["office", "desk", "work", "pc", "monitor"],
            "Living Room": ["living", "sofa", "tv", "lounge"],
            "Kitchen": ["kitchen", "dining", "cook", "oven", "fridge"],
            "Bedroom": ["bed", "sleep", "night"],
            "Exterior": ["exterior", "garden", "patio", "porch", "garage"],
            "Hallway": ["hall", "corridor", "stairs"]
        }
        
        for dev in devices:
            alias = dev['alias'].lower()
            assigned = False
            for room, keys in keywords.items():
                if any(k in alias for k in keys):
                    if room not in self.room_groups:
                        self.room_groups[room] = []
                    self.room_groups[room].append(dev)
                    assigned = True
                    break # Assign to first matching room
            
            if not assigned:
                if "Other" not in self.room_groups:
                    self.room_groups["Other"] = []
                self.room_groups["Other"].append(dev)
                
        # Update UI components
        self._update_filters()
        self._filter_grid("All")
