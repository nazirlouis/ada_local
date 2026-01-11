"""
Pocket AI - Main Entry Point
"""

import warnings
import sys

# Suppress ALL warnings globally before any other imports
# This is aggressive but ensures clean console output
warnings.simplefilter("ignore")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QColor, QIcon
from gui.app import MainWindow
from qfluentwidgets import qconfig, Theme, SplashScreen

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Configure Aura Theme
    qconfig.theme = Theme.DARK
    
    # Set default font
    app.setFont(QFont("Segoe UI", 10))
    
    # Create SplashScreen
    splash = SplashScreen(QIcon("gui/assets/logo.png" if "gui/assets/logo.png" else None), None)
    splash.setIconSize(QSize(100, 100))
    splash.show()
    
    # Create main window
    window = MainWindow()
    
    # Show window and finish splash
    window.show()
    splash.finish()
    
    sys.exit(app.exec())
