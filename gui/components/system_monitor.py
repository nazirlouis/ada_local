"""
System Monitor Component - Displays CPU, RAM, GPU usage and running Ollama models.
"""

import psutil
import requests
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import QTimer, Qt, QObject, Signal, QThread
from PySide6.QtGui import QFont

from config import OLLAMA_URL

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class MonitorWorker(QObject):
    """Worker to collect system stats in the background."""
    stats_updated = Signal(dict)

    def __init__(self):
        super().__init__()

    def collect(self):
        """Collect and emit stats."""
        try:
            stats = {}
            
            # CPU
            stats['cpu'] = psutil.cpu_percent(interval=None)
            
            # RAM
            ram = psutil.virtual_memory()
            stats['ram'] = {
                'percent': ram.percent,
                'used': ram.used / (1024 ** 3),
                'total': ram.total / (1024 ** 3)
            }
            
            # GPU
            if GPU_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    stats['gpu'] = {
                        'percent': util.gpu,
                        'vram_used': mem_info.used / (1024 ** 3),
                        'vram_total': mem_info.total / (1024 ** 3),
                        'vram_percent': (mem_info.used / mem_info.total) * 100
                    }
                except Exception:
                    stats['gpu'] = None
            else:
                stats['gpu'] = None

            # Ollama Models
            try:
                response = requests.get(f"{OLLAMA_URL}/ps", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    if models:
                        model_names = [m.get("name", "?").split(":")[0] for m in models]
                        stats['models'] = model_names
                    else:
                        stats['models'] = []
                else:
                    stats['models'] = "Offline"
            except Exception:
                stats['models'] = "Offline"

            self.stats_updated.emit(stats)
        except Exception as e:
            print(f"MonitorWorker Error: {e}")


class SystemMonitor(QFrame):
    """
    A status bar showing system resource usage and running models.
    Updates every 3 seconds via background thread.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("systemMonitor")
        self._setup_ui()
        self._init_worker()
    
    def _setup_ui(self):
        """Build the monitor UI."""
        self.setFixedHeight(32)
        self.setStyleSheet("""
            QFrame#systemMonitor {
                background: rgba(20, 20, 30, 0.9);
                border-bottom: 1px solid rgba(187, 134, 252, 0.3);
            }
            QLabel {
                color: #b0b0b0;
                font-size: 11px;
                padding: 0 8px;
            }
            QLabel#valueLabel {
                color: #e0e0e0;
                font-weight: bold;
            }
            QLabel#gpuLabel {
                color: #4fc3f7;
            }
            QLabel#modelsLabel {
                color: #81c784;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(24)
        
        # CPU
        cpu_container = QHBoxLayout()
        cpu_container.setSpacing(4)
        cpu_icon = QLabel("🖥️")
        cpu_icon.setFixedWidth(20)
        cpu_container.addWidget(cpu_icon)
        cpu_label = QLabel("CPU:")
        cpu_container.addWidget(cpu_label)
        self.cpu_value = QLabel("0%")
        self.cpu_value.setObjectName("valueLabel")
        cpu_container.addWidget(self.cpu_value)
        layout.addLayout(cpu_container)
        
        # RAM
        ram_container = QHBoxLayout()
        ram_container.setSpacing(4)
        ram_icon = QLabel("💾")
        ram_icon.setFixedWidth(20)
        ram_container.addWidget(ram_icon)
        ram_label = QLabel("RAM:")
        ram_container.addWidget(ram_label)
        self.ram_value = QLabel("0%")
        self.ram_value.setObjectName("valueLabel")
        ram_container.addWidget(self.ram_value)
        layout.addLayout(ram_container)
        
        # GPU
        gpu_container = QHBoxLayout()
        gpu_container.setSpacing(4)
        gpu_icon = QLabel("🎮")
        gpu_icon.setFixedWidth(20)
        gpu_container.addWidget(gpu_icon)
        gpu_label = QLabel("GPU:")
        gpu_container.addWidget(gpu_label)
        self.gpu_value = QLabel("N/A" if not GPU_AVAILABLE else "0%")
        self.gpu_value.setObjectName("valueLabel")
        self.gpu_value.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        gpu_container.addWidget(self.gpu_value)
        layout.addLayout(gpu_container)
        
        # VRAM
        vram_container = QHBoxLayout()
        vram_container.setSpacing(4)
        vram_label = QLabel("VRAM:")
        vram_container.addWidget(vram_label)
        self.vram_value = QLabel("N/A" if not GPU_AVAILABLE else "0 GB")
        self.vram_value.setObjectName("valueLabel")
        self.vram_value.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        vram_container.addWidget(self.vram_value)
        layout.addLayout(vram_container)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setStyleSheet("background: rgba(255,255,255,0.2);")
        separator.setFixedWidth(1)
        layout.addWidget(separator)
        
        # Running Models
        models_container = QHBoxLayout()
        models_container.setSpacing(4)
        models_icon = QLabel("🤖")
        models_icon.setFixedWidth(20)
        models_container.addWidget(models_icon)
        models_label = QLabel("Models:")
        models_container.addWidget(models_label)
        self.models_value = QLabel("Loading...")
        self.models_value.setObjectName("modelsLabel")
        self.models_value.setStyleSheet("color: #81c784; font-weight: bold;")
        models_container.addWidget(self.models_value)
        layout.addLayout(models_container)
        
        layout.addStretch()
    
    def _init_worker(self):
        """Initialize the background worker and thread."""
        self.monitor_thread = QThread()
        self.worker = MonitorWorker()
        self.worker.moveToThread(self.monitor_thread)
        
        self.worker.stats_updated.connect(self._on_stats_updated)
        
        # Use a timer to trigger collection periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.worker.collect)
        self.timer.start(3000) # 3 seconds
        
        self.monitor_thread.start()
        
        # Initial call
        QTimer.singleShot(100, self.worker.collect)

    def _on_stats_updated(self, stats):
        """Update UI with new stats from worker."""
        # CPU
        cpu_val = stats.get('cpu', 0)
        self.cpu_value.setText(f"{cpu_val:.1f}%")
        self._color_by_usage(self.cpu_value, cpu_val)
        
        # RAM
        ram_data = stats.get('ram', {})
        ram_percent = ram_data.get('percent', 0)
        self.ram_value.setText(f"{ram_percent:.1f}% ({ram_data.get('used',0):.1f}/{ram_data.get('total',0):.1f} GB)")
        self._color_by_usage(self.ram_value, ram_percent)
        
        # GPU
        gpu_data = stats.get('gpu')
        if gpu_data:
            gpu_percent = gpu_data.get('percent', 0)
            self.gpu_value.setText(f"{gpu_percent}%")
            self._color_by_usage(self.gpu_value, gpu_percent)
            
            vram_text = f"{gpu_data.get('vram_used',0):.1f}/{gpu_data.get('vram_total',0):.1f} GB"
            self.vram_value.setText(vram_text)
            self._color_by_usage(self.vram_value, gpu_data.get('vram_percent', 0))
        elif not GPU_AVAILABLE:
             self.gpu_value.setText("N/A")
             self.vram_value.setText("N/A")
        else:
             self.gpu_value.setText("Error")
             self.vram_value.setText("Error")
             
        # Models
        models = stats.get('models', [])
        if isinstance(models, list):
            if models:
                # Show up to 3 models, then count
                if len(models) <= 3:
                    self.models_value.setText(", ".join(models))
                else:
                    self.models_value.setText(f"{', '.join(models[:2])} +{len(models)-2}")
            else:
                self.models_value.setText("None")
        else:
            self.models_value.setText(models) # "Offline" or "Loading..."

    def _color_by_usage(self, label: QLabel, percent: float):
        """Color the label based on usage percentage."""
        if percent >= 90:
            color = "#ef5350"  # Red
        elif percent >= 70:
            color = "#ffb74d"  # Orange
        elif percent >= 50:
            color = "#fff176"  # Yellow
        else:
            color = "#81c784"  # Green
        label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def __del__(self):
        """Cleanup thread."""
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.quit()
            self.monitor_thread.wait()
