from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QScrollArea, QGridLayout, QPushButton
)
from PySide6.QtCore import Qt, QTimer, QDate, QTime, QSize, Signal, QThread
from PySide6.QtGui import QFont, QColor

from qfluentwidgets import (
    CardWidget, TitleLabel, BodyLabel, StrongBodyLabel, 
    FluentIcon as FIF, IconWidget, TransparentToolButton,
    SimpleCardWidget, ImageLabel, PillToolButton
)

from core.news import news_manager
from core.tasks import task_manager
from core.calendar import calendar_manager
from core.kasa_control import kasa_manager
from datetime import datetime, timedelta
import asyncio

# --- Components ---

from core.weather import weather_manager

class GreetingsHeader(QWidget):
    """
    Header showing "Good [Morning/Afternoon/Evening]" and Bubbles (Time | Weather).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 20)
        self.layout.setSpacing(15)
        
        # Text Block
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        self.sub_label = BodyLabel("Welcome back, User")
        self.sub_label.setStyleSheet("color: #8b9bb4;")
        
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 42px; font-weight: bold; color: #33b5e5; font-family: 'Segoe UI', sans-serif;")
        
        self.date_label = BodyLabel()
        self.date_label.setStyleSheet("color: #8b9bb4; font-size: 14px;")
        
        text_layout.addWidget(self.sub_label)
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.date_label)
        
        self.layout.addLayout(text_layout)
        self.layout.addStretch()
        
        # --- Bubbles Container ---
        # We use a container to hold the two bubbles side-by-side or grouped
        bubbles_layout = QHBoxLayout()
        bubbles_layout.setSpacing(15)
        
        # 1. Time Bubble
        self.time_bubble = QFrame()
        self.time_bubble.setFixedSize(160, 100)
        self.time_bubble.setStyleSheet("""
            QFrame {
                background-color: #0f1524;
                border: 1px solid #1a2236;
                border-radius: 20px;
            }
        """)
        tb_layout = QVBoxLayout(self.time_bubble)
        tb_layout.setAlignment(Qt.AlignCenter)
        
        self.clock_label = QLabel()
        self.clock_label.setStyleSheet("color: #e8eaed; font-size: 28px; font-weight: bold;")
        tb_layout.addWidget(self.clock_label)
        
        bubbles_layout.addWidget(self.time_bubble)
        
        # 2. Weather Bubble
        self.weather_bubble = QFrame()
        self.weather_bubble.setFixedSize(160, 100)
        self.weather_bubble.setStyleSheet("""
            QFrame {
                background-color: #0f1524;
                border: 1px solid #1a2236;
                border-radius: 20px;
            }
        """)
        wb_layout = QVBoxLayout(self.weather_bubble)
        wb_layout.setAlignment(Qt.AlignCenter)
        wb_layout.setSpacing(5)
        
        # Icon Area
        # Using a label for emoji or IconWidget
        self.w_icon_label = QLabel("☁️") # Default emoji
        self.w_icon_label.setStyleSheet("font-size: 24px; background: transparent;")
        self.w_icon_label.setAlignment(Qt.AlignCenter)
        
        self.temp_label = QLabel("--°F")
        self.temp_label.setStyleSheet("color: #e8eaed; font-size: 18px; font-weight: bold;")
        
        self.cond_label = QLabel("Loading...")
        self.cond_label.setStyleSheet("color: #8b9bb4; font-size: 12px;")
        
        wb_layout.addWidget(self.w_icon_label)
        wb_layout.addWidget(self.temp_label)
        wb_layout.addWidget(self.cond_label)
        
        bubbles_layout.addWidget(self.weather_bubble)
        
        self.layout.addLayout(bubbles_layout)
        
        # Timer
        self._update_time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000) 
        
        # Weather update (async would be better, but simple timer fits pattern)
        # Fetch immediately then every 15 mins
        self._fetch_weather()
        self.w_timer = QTimer(self)
        self.w_timer.timeout.connect(self._fetch_weather)
        self.w_timer.start(900000) 

    def _update_time(self):
        now = datetime.now()
        hour = now.hour
        
        if 5 <= hour < 12:
            greeting = "Good Morning"
        elif 12 <= hour < 18:
            greeting = "Good Afternoon"
        else:
            greeting = "Good Evening"
            
        self.title_label.setText(greeting)
        self.date_label.setText("📅 " + QDate.currentDate().toString("dddd, MMMM d"))
        self.clock_label.setText(QTime.currentTime().toString("h:mm AP"))

    def _fetch_weather(self):
        # Run in thread to avoid freeze
        self._thread = QThread()
        self._worker = WeatherWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_weather_loaded)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_weather_loaded(self, data):
        if not data:
            self.cond_label.setText("Offline")
            return
            
        temp = data['temp']
        code = data['code']
        
        self.temp_label.setText(f"{int(temp)}°F")
        
        # Mapping Code to Emoji/Text
        # 0: Clear (☀️), 1-3: Cloud (☁️), 45+: Cloud/Fog, 51+: Rain (🌧️), 71+: Snow (❄️), 95+: Storm (⚡)
        if code == 0:
            icon, text = "☀️", "Clear"
        elif code in [1, 2, 3]:
            icon, text = "⛅", "Cloudy"
        elif code in [45, 48]:
            icon, text = "🌫️", "Foggy"
        elif code in [51, 53, 55, 61, 63, 65]:
            icon, text = "🌧️", "Rain"
        elif code in [71, 73, 75, 85, 86]:
            icon, text = "❄️", "Snow"
        elif code >= 95:
             icon, text = "⚡", "Storm"
        else:
             icon, text = "🌡️", "Unknown"
             
        self.w_icon_label.setText(icon)
        self.cond_label.setText(text)

class WeatherWorker(QThread):
    finished = Signal(dict)
    def run(self):
        data = weather_manager.get_weather()
        self.finished.emit(data or {})


class StatCard(CardWidget):
    """
    Square/Rectangular card for quick stats (Agenda, Devices, etc).
    Clickable - emits clicked signal with route_key for navigation.
    """
    clicked = Signal(str)
    
    def __init__(self, icon: FIF, title: str, count: str, route_key: str = None, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 110)
        self.setBorderRadius(16)
        self.route_key = route_key
        self.setCursor(Qt.PointingHandCursor) if route_key else None
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 25, 20)
        
        # Left Side: Icon + Title
        left = QVBoxLayout()
        left.setSpacing(15)
        
        # Icon bubble
        icon_bubble = QLabel()
        icon_bubble.setFixedSize(40, 40)
        icon_bubble.setAlignment(Qt.AlignCenter)
        # Using a QFrame approach or just styling the label
        # Since I can't easily embed FluentIcon in stylesheet, use IconWidget in a container
        
        ib_container = QFrame()
        ib_container.setFixedSize(40, 40)
        ib_container.setStyleSheet("background-color: #1a2236; border-radius: 12px;")
        ib_layout = QVBoxLayout(ib_container)
        ib_layout.setContentsMargins(0,0,0,0)
        ib_layout.setAlignment(Qt.AlignCenter)
        
        iw = IconWidget(icon)
        iw.setFixedSize(20, 20)
        # Tint icon
        # iw.setStyleSheet("color: #33b5e5;") # IconWidget doesn't style this way easily
        ib_layout.addWidget(iw)
        
        left.addWidget(ib_container)
        
        lbl = BodyLabel(title)
        lbl.setStyleSheet("color: #8b9bb4; font-size: 13px; font-weight: 500;")
        left.addWidget(lbl)
        
        layout.addLayout(left)
        layout.addStretch()
        
        # Right Side: Big Number
        self.num_label = QLabel(str(count))
        self.num_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #e8eaed;")
        self.num_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(self.num_label)

    def set_count(self, count):
        self.num_label.setText(str(count))
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.route_key:
            self.clicked.emit(self.route_key)

class HomeScenesCard(CardWidget):
    """
    Card with Home Scene buttons that control Kasa devices.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 160)
        self.setBorderRadius(16)
        self._devices = []
        self._action_thread = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        t1 = StrongBodyLabel("Home Scenes")
        t2 = BodyLabel("Instant environmental adjustments.")
        t2.setStyleSheet("color: #6e7a8e; font-size: 12px;")
        
        layout.addWidget(t1)
        layout.addWidget(t2)
        
        # Buttons
        btns = QHBoxLayout()
        self.focus_btn = QPushButton("Focus Mode")
        self.focus_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a2236; color: #e8eaed; border: 1px solid #1a2236; 
                border-radius: 8px; padding: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #232d45; }
        """)
        self.focus_btn.clicked.connect(self._on_focus_mode)
        
        self.relax_btn = QPushButton("Relax")
        self.relax_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent; color: #e8eaed; border: 1px solid #1a2236; 
                border-radius: 8px; padding: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1a2236; }
        """)
        self.relax_btn.clicked.connect(self._on_relax_mode)
        
        btns.addWidget(self.focus_btn)
        btns.addWidget(self.relax_btn)
        layout.addLayout(btns)
    
    def set_devices(self, devices: list):
        """Store the discovered devices for scene control."""
        self._devices = devices
    
    def _on_focus_mode(self):
        """Focus Mode: Turn off all lights to minimize distractions."""
        if not self._devices:
            return
        self._run_scene_action(self._focus_action)
    
    def _on_relax_mode(self):
        """Relax Mode: Dim lights to 40%."""
        if not self._devices:
            return
        self._run_scene_action(self._relax_action)
    
    def _run_scene_action(self, action_func):
        """Run a scene action in a background thread."""
        class SceneThread(QThread):
            def __init__(self, func):
                super().__init__()
                self.func = func
            def run(self):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.func())
                    loop.close()
                except Exception as e:
                    print(f"Scene action error: {e}")
        
        # Wait for any previous thread to finish (with safety check)
        try:
            if self._action_thread is not None and self._action_thread.isRunning():
                self._action_thread.wait()
        except RuntimeError:
            # C++ object already deleted, that's fine
            pass
        
        self._action_thread = SceneThread(action_func)
        # Clear our reference when finished to avoid stale reference issues
        self._action_thread.finished.connect(lambda: setattr(self, '_action_thread', None))
        self._action_thread.start()
    
    async def _focus_action(self):
        """Turn off all discovered devices."""
        for device in self._devices:
            try:
                await kasa_manager.turn_off(device['ip'])
            except Exception as e:
                print(f"Focus mode error for {device['alias']}: {e}")
    
    async def _relax_action(self):
        """Dim all dimmable devices to 40%."""
        for device in self._devices:
            try:
                if device.get('brightness') is not None:
                    await kasa_manager.set_brightness(device['ip'], 40)
                    await kasa_manager.turn_on(device['ip'])
                else:
                    # For non-dimmable, just turn on
                    await kasa_manager.turn_on(device['ip'])
            except Exception as e:
                print(f"Relax mode error for {device['alias']}: {e}")


class IntelligenceItem(QFrame):
    """
    Single row in the Intelligence Feed.
    """
    def __init__(self, icon: FIF, title: str, description: str, time_str: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(15)
        
        # Icon
        ic_bg = QFrame()
        ic_bg.setFixedSize(36, 36)
        ic_bg.setStyleSheet("background-color: #141c2f; border-radius: 10px;")
        icl = QVBoxLayout(ic_bg)
        icl.setContentsMargins(0,0,0,0)
        icl.setAlignment(Qt.AlignCenter)
        icl.addWidget(IconWidget(icon))
        
        layout.addWidget(ic_bg)
        
        # Content
        col = QVBoxLayout()
        col.setSpacing(4)
        
        top_line = QHBoxLayout()
        self.t_lbl = StrongBodyLabel(title)
        self.time_lbl = BodyLabel(time_str)
        self.time_lbl.setStyleSheet("color: #6e7a8e; font-size: 11px;")
        
        top_line.addWidget(self.t_lbl)
        top_line.addStretch()
        top_line.addWidget(self.time_lbl)
        
        self.desc_lbl = BodyLabel(description)
        self.desc_lbl.setStyleSheet("color: #8b9bb4; font-size: 12px;")
        self.desc_lbl.setWordWrap(True)
        
        col.addLayout(top_line)
        col.addWidget(self.desc_lbl)
        
        layout.addLayout(col)
        
    def update_content(self, title: str, description: str, time_str: str):
        self.t_lbl.setText(title)
        self.desc_lbl.setText(description)
        self.time_lbl.setText(time_str)

class IntelligenceFeed(CardWidget):
    """
    Main content area: System Intelligence.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBorderRadius(20)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(30, 25, 30, 30)
        self.layout.setSpacing(20)
        
        # Header
        h_layout = QHBoxLayout()
        h = TitleLabel("System Intelligence", self)
        h.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        live_tag = QLabel("Live Updates")
        live_tag.setStyleSheet("""
            background-color: #1a2236; color: #8b9bb4; 
            padding: 4px 10px; border-radius: 6px; font-size: 10px; font-weight: bold;
        """)
        
        h_layout.addWidget(h)
        h_layout.addStretch()
        h_layout.addWidget(live_tag)
        
        self.layout.addLayout(h_layout)
        
        # 1. Daily Focus
        self.focus_item = IntelligenceItem(
            FIF.TILES,
            "Daily Focus",
            "Analyzing your schedule...",
            "NOW"
        )
        self.layout.addWidget(self.focus_item)
        
        # Separator
        sep1 = QFrame()
        sep1.setFixedHeight(1)
        sep1.setStyleSheet("background-color: #1a2236;")
        self.layout.addWidget(sep1)
        
        # 2. News (Intel Alert)
        self.news_item = IntelligenceItem(
            FIF.WIFI,
            "Intel Alert",
            "Syncing intelligence streams...",
            "NOW"
        )
        self.layout.addWidget(self.news_item)
            
        # Separator
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background-color: #1a2236;")
        self.layout.addWidget(sep2)
        
        # 3. Devices Status
        self.devices_item = IntelligenceItem(
            FIF.IOT, 
            "Smart Home",
            "Scanning for connected devices...",
            "NOW"
        )
        self.layout.addWidget(self.devices_item)
        
        self.layout.addStretch()
        
        # Upcoming Priority Card (Embedded at bottom)
        self.priority = PriorityCard()
        self.layout.addWidget(self.priority)

    def update_news(self, news):
        if news:
            top = news[0]
            self.news_item.update_content("Intel Alert", top['title'], "JUST NOW")
        else:
            self.news_item.update_content("Intel Alert", "No active intelligence streams detected.", "NOW")
    
    def update_devices(self, devices):
        count = len(devices) if devices else 0
        online = sum(1 for d in devices if d.get('is_on')) if devices else 0
        if count > 0:
            self.devices_item.update_content(
                "Smart Home", 
                f"{count} devices found, {online} currently on.", 
                "LIVE"
            )
        else:
            self.devices_item.update_content(
                "Smart Home", 
                "No Kasa devices detected on network.", 
                "NOW"
            )
    
    def update_focus(self, tasks):
        active = [t for t in tasks if not t.get('completed')]
        if active:
            self.focus_item.update_content(
                "Daily Focus",
                f"You have {len(active)} active task{'s' if len(active) != 1 else ''} to complete today.",
                "NOW"
            )
        else:
            self.focus_item.update_content(
                "Daily Focus",
                "All tasks completed. Great job!",
                "NOW"
            )

class PriorityCard(QFrame):
    """
    Blue gradient card for next priority.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #33b5e5, stop:1 #4a68af);
                border-radius: 16px;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(25, 0, 25, 0)
        
        # Text
        txt = QVBoxLayout()
        txt.setAlignment(Qt.AlignVCenter)
        txt.setSpacing(5)
        
        l1 = QLabel("Upcoming Priority")
        l1.setStyleSheet("color: white; font-weight: bold; font-size: 16px; background: transparent;")
        
        self.title_label = QLabel("No upcoming events")
        self.title_label.setStyleSheet("color: rgba(255,255,255,0.9); font-size: 14px; background: transparent;")
        
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 12px; background: transparent;")
        
        txt.addWidget(l1)
        txt.addWidget(self.title_label)
        txt.addWidget(self.time_label)
        
        layout.addLayout(txt)
        layout.addStretch()
        
        # Button
        btn = QPushButton("Details")
        btn.setFixedSize(80, 32)
        btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.3);
                font-weight: bold;
            }
            QPushButton:hover { background-color: rgba(255,255,255,0.3); }
        """)
        layout.addWidget(btn)

    def update_event(self, events: list):
        """Update with the next upcoming event from the list."""
        now = datetime.now()
        next_event = None
        
        for event in events:
            try:
                start_time = datetime.strptime(event['start_time'], "%Y-%m-%d %H:%M:%S")
                if start_time > now:
                    if next_event is None or start_time < datetime.strptime(next_event['start_time'], "%Y-%m-%d %H:%M:%S"):
                        next_event = event
            except (KeyError, ValueError):
                continue
        
        if next_event:
            self.title_label.setText(next_event['title'])
            start_time = datetime.strptime(next_event['start_time'], "%Y-%m-%d %H:%M:%S")
            delta = start_time - now
            minutes = int(delta.total_seconds() / 60)
            if minutes < 60:
                self.time_label.setText(f"Starts in {minutes} minutes")
            else:
                hours = minutes // 60
                self.time_label.setText(f"Starts in {hours} hour{'s' if hours > 1 else ''}")
        else:
            self.title_label.setText("No upcoming events")
            self.time_label.setText("Enjoy your free time!")

class DashboardLoader(QThread):
    finished = Signal(dict)
    
    def run(self):
        # Fetch tasks, news, devices, and calendar in background
        try:
            tasks = task_manager.get_tasks()
            news = news_manager.get_briefing(use_ai=False)
            
            # Fetch Kasa devices
            devices = []
            try:
                print("[Dashboard] Starting Kasa device discovery...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                devices = loop.run_until_complete(kasa_manager.discover_devices())
                loop.close()
                print(f"[Dashboard] Found {len(devices)} devices")
            except Exception as e:
                print(f"[Dashboard] Kasa discovery error: {e}")
            
            # Fetch today's calendar events
            today_str = datetime.now().strftime("%Y-%m-%d")
            events = calendar_manager.get_events(today_str)
            
            print("[Dashboard] Data loading complete, emitting signal")
            self.finished.emit({
                "tasks": tasks,
                "news": news,
                "devices": devices,
                "events": events
            })
        except Exception as e:
            print(f"[Dashboard] Loader error: {e}")
            self.finished.emit({"tasks": [], "news": [], "devices": [], "events": []})

class DashboardView(QWidget):
    """
    The main 'System Intelligence' Dashboard.
    Emits navigate_to signal when user clicks on a stat card to go to that tab.
    """
    navigate_to = Signal(str)  # Emits the route key (e.g., "plannerInterface")
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dashboardView")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        # 1. Greeting Header
        self.header = GreetingsHeader()
        main_layout.addWidget(self.header)
        
        # 2. Main Content Area (2 Columns)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)
        
        # --- Left Column (Stats) ---
        left_col = QVBoxLayout()
        left_col.setSpacing(20)
        
        # Stat 1: Planner - navigates to plannerInterface
        self.planner_stat = StatCard(FIF.CALENDAR, "Planner Agenda", "--", "plannerInterface")
        self.planner_stat.clicked.connect(self._on_navigate)
        left_col.addWidget(self.planner_stat)
        
        # Stat 2: Devices - navigates to homeInterface
        self.devices_stat = StatCard(FIF.IOT, "Active Devices", "--", "homeInterface")
        self.devices_stat.clicked.connect(self._on_navigate)
        left_col.addWidget(self.devices_stat)
        
        # Stat 3: Unread News - navigates to briefingInterface
        self.news_stat = StatCard(FIF.TILES, "Unread News", "--", "briefingInterface")
        self.news_stat.clicked.connect(self._on_navigate)
        left_col.addWidget(self.news_stat)
        
        # Home Scenes
        self.home_scenes = HomeScenesCard()
        left_col.addWidget(self.home_scenes)
        
        left_col.addStretch()
        content_layout.addLayout(left_col)
        
        # --- Right Column (Feed) ---
        self.feed = IntelligenceFeed()
        # Feed card should expand
        content_layout.addWidget(self.feed, 1)
        
        main_layout.addLayout(content_layout)
        
        # Store devices for scene control
        self._devices = []
        self.loader = None
        
        # Trigger async load
        QTimer.singleShot(100, self._start_loading)

    def _start_loading(self):
        # Prevent multiple loaders
        if self.loader and self.loader.isRunning():
            return
        self.loader = DashboardLoader(self)
        self.loader.finished.connect(self._on_data_loaded)
        self.loader.finished.connect(self.loader.deleteLater)
        self.loader.start()

    def _on_data_loaded(self, data):
        tasks = data.get("tasks", [])
        news = data.get("news", [])
        devices = data.get("devices", [])
        events = data.get("events", [])
        
        # Store devices for scene control
        self._devices = devices
        self.home_scenes.set_devices(devices)
        
        # Update stat cards
        active_tasks = [t for t in tasks if not t.get('completed')]
        self.planner_stat.set_count(len(active_tasks))
        self.news_stat.set_count(len(news))
        self.devices_stat.set_count(len(devices))
        
        # Update intelligence feed
        self.feed.update_news(news)
        self.feed.update_devices(devices)
        self.feed.update_focus(tasks)
        
        # Update priority card with calendar events
        self.feed.priority.update_event(events)
    
    def _on_navigate(self, route_key: str):
        """Emit navigation signal when a stat card is clicked."""
        self.navigate_to.emit(route_key)
