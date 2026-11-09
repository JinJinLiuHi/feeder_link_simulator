"""The startup screen prepares all simulation pages before revealing the UI."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

import time
import unittest
from unittest.mock import patch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QTabWidget
from constellation_sim.ui.main_window import MainWindow
from constellation_sim.ui.startup import StartupWindow
from link_sim.ui.workspace import LazyDialogPage


class StartupPreloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_all_pages_ready_before_main_window_opens(self):
        splash = StartupWindow()
        opened = []
        errors = []

        def map_ready(window):
            # Map loading starts only once every link page is initialized.
            self.assertTrue(all(p.loaded for p in window.findChildren(LazyDialogPage)))
            QTimer.singleShot(0, lambda: window.web.loadFinished.emit(True))

        def reveal(window):
            opened.append(window)
            self.assertEqual(window._active_primary_workspace, "constellation")
            pages = window.findChildren(LazyDialogPage)
            self.assertEqual(len(pages), 7)
            self.assertTrue(all(p.loaded and not p.loading for p in pages))

        try:
            with patch.object(MainWindow, "load_map", map_ready), \
                 patch.object(MainWindow, "showMaximized", reveal), \
                 patch.object(splash, "fail", side_effect=errors.append):
                splash.build_main_window(MainWindow)
                deadline = time.monotonic()+30
                while not splash._finished and not errors and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertEqual(errors, [])
                self.assertTrue(splash._finished)
                self.assertEqual(opened, [splash.main_window])
                splash.finish(True)
                self.assertEqual(len(opened), 1)
                # Subsequent navigation must reuse the startup-created editors.
                with patch("link_sim.ui.workspace._load_class", side_effect=AssertionError("unexpected late import")):
                    for i in range(3):
                        splash.main_window._switch_primary_workspace("simulation", i, True)
                    channel = splash.main_window.view_tabs._stack.widget(
                        splash.main_window._workspace_groups["simulation"][0])
                    tabs = channel.findChild(QTabWidget)
                    for i in range(tabs.count()):
                        tabs.setCurrentIndex(i)
                        tabs.widget(i).ensure_loaded()
                    self.app.processEvents()
        finally:
            splash.map_timeout.stop()
            if splash.main_window is not None:
                splash.main_window.close()
                splash.main_window.deleteLater()
            splash.close()
            splash.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
