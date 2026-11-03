"""Readable library selection, Beijing timestamps and computed ADC density."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from PySide6.QtWidgets import QApplication, QPlainTextEdit

from link_sim.algorithm.device_models import DeviceModel
from link_sim.algorithm.model_store import ModelStore, format_beijing_time
from link_sim.algorithm.adc_noise_core import calibrate_model_noise
from link_sim.algorithm.waveform_chain import Stage
from link_sim.ui.model_library import ModelLibraryDialog, SaveModelDialog
from link_sim.ui.waveform_workspace import StageDialog
from link_sim.algorithm.ad_model import ADDAModelDialog
from link_sim.ui.device_model_bindings import apply_device


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

    def test_new_records_use_beijing_offset_and_library_has_three_columns(self):
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
                self.assertEqual(dialog.table.columnCount(), 3)
                self.assertEqual([dialog.table.horizontalHeaderItem(i).text() for i in range(3)],
                                 ["名称", "器件类型", "保存时间（北京时间）"])
                self.assertEqual(dialog.table.item(0, 2).text(), "2026-10-28 11:51:14")
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
