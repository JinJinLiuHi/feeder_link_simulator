"""Navigation text and panel widths must survive show, resize and page switches."""

import os
# Qt WebEngine in the main window needs the Windows platform plugin for clean
# teardown; the pure Qt waveform tests can still run offscreen independently.
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

import unittest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from constellation_sim.ui.main_window import MainWindow


class NavigationLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.window = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.window.deleteLater()
        cls.app.processEvents()

    def test_expanded_navigation_survives_window_layout_enforcement(self):
        w = self.window
        w._switch_primary_workspace("simulation", 0, True)
        self.app.processEvents()
        for width in (1000, 1500):
            w.resize(width, 850)
            w._enforce_splitter_layout()
            self.app.processEvents()
            self.assertEqual(w.project_panel.width(), w.project_rail.width())
            self.assertEqual(w.project_rail.width(), 168)
        for button in w.project_buttons + w._secondary_workspace_buttons["simulation"]:
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.assertLessEqual(button.sizeHint().width(), button.width())

    def test_collapsing_and_closing_drawer_preserves_active_workspace(self):
        w = self.window
        w._switch_primary_workspace("simulation", 0, True)
        w._set_project_drawer(True)
        self.assertGreater(w.project_panel.width(), w.project_rail.width())
        w._switch_primary_workspace("simulation", 1, True)
        self.app.processEvents()
        self.assertFalse(w._drawer_expanded(w.project_panel))
        self.assertEqual(w.project_panel.width(), 168)
        self.assertTrue(w.project_buttons[1].isChecked())
        w._set_secondary_navigation(False)
        w._enforce_splitter_layout()
        self.app.processEvents()
        self.assertEqual(w.project_rail.width(), 48)
        self.assertEqual(w.project_panel.width(), 48)
        for button in w.project_buttons:
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.assertEqual(len(button.text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
