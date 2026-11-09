"""Readable library selection, Beijing timestamps and computed ADC density."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QMessageBox, QDialog

from link_sim.algorithm.device_models import DeviceModel
from link_sim.algorithm.model_store import ModelStore, format_beijing_time
from link_sim.algorithm.adc_noise_core import calibrate_model_noise
from link_sim.algorithm.waveform_chain import Stage
from link_sim.ui.model_library import ModelLibraryDialog, SaveModelDialog
from link_sim.ui.waveform_workspace import StageDialog
from link_sim.algorithm.ad_model import ADDAModelDialog
from link_sim.ui.device_model_bindings import apply_device
from link_sim.ui.workspace import LazyDialogPage


def device():
    return DeviceModel("adda", dict(n_bits=12, fs=3e9, nfft=65536, f_sig=500e6,
        noise_density_dbfs_hz=-151.8, sfdr_target=60., device_profile="ADC12DJ3200",
        spur_freqs=[1.2e9], spur_dbfs=[-60.]), name="ADC12DJ3200", description="hidden description").validated()


class ModelLibraryDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_beijing_format_including_utc_history_and_day_rollover(self):
        self.assertEqual(format_beijing_time("2026-10-28T03:51:14+00:00"), "2026-10-28 11:51:14")
        self.assertEqual(format_beijing_time("2026-11-03T18:04:05Z"), "2026-11-04 02:04:05")
        self.assertEqual(format_beijing_time("2026-11-04T02:04:05+08:00"), "2026-11-04 02:04:05")
        self.assertEqual(format_beijing_time("2026-11-03T18:04:05"), "2026-11-04 02:04:05")

    def test_library_shows_stable_id_before_name_and_beijing_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            key = store.save(device())
            entry = store.list()[0]
            self.assertEqual(datetime.fromisoformat(entry["created_at"]).utcoffset(), timedelta(hours=8))
            # Existing UTC records must display in Beijing time without altering their instant.
            with store.connection() as db:
                db.execute("UPDATE models SET created_at=? WHERE id=?", ("2026-10-28T03:51:14+00:00", key))
            dialog = ModelLibraryDialog(store=store)
            try:
                self.assertEqual(dialog.table.columnCount(), 4)
                self.assertEqual([dialog.table.horizontalHeaderItem(i).text() for i in range(4)],
                                 ["编号", "名称", "器件类型", "保存时间（北京时间）"])
                self.assertEqual(dialog.table.item(0, 0).text(), str(key))
                self.assertEqual(dialog.table.item(0, 1).text(), device().name)
                self.assertEqual(dialog.table.item(0, 3).text(), "2026-10-28 11:51:14")
                self.assertEqual(dialog.selected_id, key)
                self.assertNotIn("hidden description", dialog.details.toPlainText())
                self.assertIn("#12617B", dialog.styleSheet())
                self.assertNotIn("background-color: #E6F1FB", dialog.styleSheet())
                dialog.load_selected()
                self.assertEqual(dialog.selected_model.parameters, device().parameters)
            finally:
                dialog.close()
                dialog.deleteLater()
                self.app.processEvents()

    def test_delete_confirmation_refresh_and_search_by_stable_id(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            first = store.save(device())
            model = device()
            model.name = "另一模型"
            second = store.save(model)
            dialog = ModelLibraryDialog(store=store)
            try:
                self.assertEqual(dialog.selected_id, second)
                self.assertTrue(dialog.delete_button.isEnabled())
                with patch("link_sim.ui.model_library.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
                    dialog.delete_button.click()
                self.assertEqual(len(store.list()), 2)
                with patch("link_sim.ui.model_library.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes) as ask:
                    dialog.delete_button.click()
                    self.assertIn(f"#{second}", ask.call_args.args[2])
                    self.assertEqual(dialog.selected_id, first)
                    self.assertEqual(dialog.table.rowCount(), 1)
                    dialog.delete_button.click()
                self.assertEqual(dialog.table.rowCount(), 0)
                self.assertIsNone(dialog.selected_id)
                self.assertFalse(dialog.delete_button.isEnabled())
                self.assertFalse(dialog.load_button.isEnabled())
                self.assertFalse(dialog.export_button.isEnabled())
                new_key = store.save(device())
                self.assertGreater(new_key, second)
                store.save(model)
                dialog.search.setText(f"#{new_key}")
                self.assertEqual(dialog.table.rowCount(), 1)
                self.assertEqual(dialog.selected_id, new_key)
                self.assertIn(f"#{new_key} ", dialog.details.toPlainText())
            finally:
                dialog.close()
                dialog.deleteLater()
                self.app.processEvents()

    def test_embedded_library_double_click_and_details_button_do_not_hide_page(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            key = store.save(device())
            page = LazyDialogPage("link_sim.ui.model_library", "ModelLibraryDialog")
            page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
            try:
                with patch("link_sim.ui.model_library.current_store", return_value=store):
                    page.ensure_loaded()
                page.show()
                self.app.processEvents()
                dialog = page.dialog
                self.assertFalse(dialog.isWindow())
                dialog.table.itemDoubleClicked.emit(dialog.table.item(0, 1))
                self.app.processEvents()
                self.assertTrue(dialog.isVisible())
                self.assertTrue(dialog.table.isVisible())
                self.assertEqual(dialog.selected_id, key)
                self.assertIn(device().name, dialog.details.toPlainText())
                dialog.load_button.click()
                self.assertTrue(dialog.isVisible())
                self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
            finally:
                page.close()
                page.deleteLater()
                self.app.processEvents()

    def test_typed_picker_double_click_still_loads_model_and_closes_dialog(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            key = store.save(device())
            dialog = ModelLibraryDialog(kind="adda", store=store)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
            try:
                dialog.show()
                self.app.processEvents()
                dialog.table.itemDoubleClicked.emit(dialog.table.item(0, 1))
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
                self.assertFalse(dialog.isVisible())
                self.assertEqual(dialog.selected_id, key)
            finally:
                dialog.close()
                dialog.deleteLater()
                self.app.processEvents()

    def test_save_dialog_only_requires_name(self):
        dialog = SaveModelDialog(device())
        try:
            self.assertFalse(dialog.findChildren(QPlainTextEdit))
            self.assertFalse(hasattr(dialog, "description"))
            self.assertEqual(dialog.name.text(), "ADC12DJ3200")
        finally:
            dialog.close()
            dialog.deleteLater()
            self.app.processEvents()

    def test_density_visible_in_original_adc_and_canvas_module(self):
        model = device()
        estimate = calibrate_model_noise(model)
        editor = ADDAModelDialog()
        module = StageDialog(Stage("d", "dac", "AD/DA", {}, model))
        try:
            apply_device(editor, model)
            for text in (editor.status.text(), module.details.toPlainText()):
                self.assertIn("D_n", text)
                self.assertIn(f"{estimate.density:.6e}", text)
                self.assertIn(f"{estimate.density_dbfs_hz:.4f}", text)
                self.assertIn("归一化功率/Hz", text)
            self.assertNotIn("版本", module.model_label.text())
        finally:
            editor.close()
            module.close()
            editor.deleteLater()
            module.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
