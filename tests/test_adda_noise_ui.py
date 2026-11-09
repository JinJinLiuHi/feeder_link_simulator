"""Original editor stays intact; canvas consumes its saved model output error."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QTabWidget, QPushButton, QDoubleSpinBox

from link_sim.algorithm.device_models import DeviceModel
from link_sim.algorithm.model_store import ModelStore
from link_sim.algorithm.waveform_chain import Stage, run_graph
from link_sim.ui.device_model_bindings import capture_device, apply_device
from link_sim.ui.model_library import ModelLibraryDialog
from link_sim.ui.workspace import ChannelWorkspace, LazyDialogPage
from link_sim.ui.waveform_workspace import StageDialog, WaveformWorkspace


def device():
    return DeviceModel("adda", dict(n_bits=12, fs=64e6, nfft=8192, f_sig=8e6,
        noise_density_dbfs_hz=-150., sfdr_target=60., device_profile="测试原模型",
        spur_freqs=[12e6], spur_dbfs=[-60.]), name="原 ADC 模型").validated()


class ADDAOriginalEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_original_page_and_model_binding_are_restored(self):
        channel = ChannelWorkspace()
        tabs = channel.findChild(QTabWidget)
        self.assertEqual(tabs.tabText(3), "ADC/DAC")
        page = tabs.widget(3)
        self.assertIsInstance(page, LazyDialogPage)
        page.ensure_loaded()
        try:
            editor = page.dialog
            self.assertEqual(type(editor).__name__, "ADDAModelDialog")
            self.assertTrue(hasattr(editor, "e_bits"))
            self.assertTrue(hasattr(editor, "e_sfdr_target"))
            self.assertTrue(hasattr(editor, "spur_form"))
            model = device()
            apply_device(editor, model)
            snapshot = capture_device(editor)
            self.assertEqual(snapshot.parameters, model.parameters)
            editor.e_bits.setText("10")
            self.assertEqual(capture_device(editor).parameters["n_bits"], 10)
        finally:
            channel.close()
            channel.deleteLater()
            self.app.processEvents()

    def test_canvas_loads_original_library_model_without_replacing_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"models.sqlite3")
            model = device()
            ModelStore(path).save(model)
            dialog = StageDialog(Stage("d", "dac", "AD/DA", {}))
            try:
                def chooser(parent, kind):
                    picker = ModelLibraryDialog(parent, kind)
                    QTimer.singleShot(0, picker.load_selected)
                    return picker

                with patch.dict(os.environ, {"FEEDER_MODEL_DB": path}), \
                     patch("link_sim.ui.waveform_workspace.ModelLibraryDialog", side_effect=chooser):
                    dialog.choose_model()
                self.assertEqual(dialog.stage.model.parameters, model.parameters)
                self.assertEqual(dialog.fields, {})
                self.assertFalse(dialog.findChildren(QDoubleSpinBox))
                labels = [b.text() for b in dialog.findChildren(QPushButton)]
                self.assertIn("从模型库加载…", labels)
                self.assertIn("导入模型文件…", labels)
                self.assertNotIn("恢复内置模型", labels)
                dialog.accept()
                self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            finally:
                dialog.close()
                dialog.deleteLater()
                self.app.processEvents()

    def test_canvas_observes_and_exports_the_added_error_separately(self):
        w = WaveformWorkspace()
        from waveform_fixtures import configure_workspace_rrc
        configure_workspace_rrc(w)
        try:
            node = next(n for n in w.graph.all_nodes() if n.KIND == "dac")
            from link_sim.algorithm.waveform_chain import model_to_dict
            node.set_property("device_model", model_to_dict(device()))
            source = next(n for n in w.graph.all_nodes() if n.KIND == "frame_source")
            source.set_property("settings", dict(source.get_property("settings"), fs=16e6))
            w.run_chain()
            deadline = time.monotonic()+15
            while w.worker and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertIsNone(w.worker)
            outputs = w.outputs
            index = w.point.findData(node.id+"::error")
            self.assertGreaterEqual(index, 0)
            w.point.setCurrentIndex(index)
            np.testing.assert_array_equal(w.observed_wave().x, outputs[node.id].added_error)
        finally:
            w.close()
            w.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
