"""Qt integration: actual worker, selection and embedded model configuration."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QDialog, QDoubleSpinBox, QPushButton, QToolButton

from link_sim.algorithm.device_models import from_pa
from link_sim.algorithm.pa_dpd import PADPDWidget
from link_sim.algorithm.pa_simulation_core import PAModel
from link_sim.algorithm.waveform_chain import Stage
from link_sim.ui.waveform_workspace import StageDialog, WaveformWorkspace


class WaveformWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.workspace = WaveformWorkspace()

    def tearDown(self):
        self.workspace.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    def wait_for_worker(self):
        deadline = time.monotonic()+15
        while self.workspace.worker and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(self.workspace.worker, "Background computation did not finish")
        self.assertEqual(len(self.workspace.outputs), 6)

    def test_device_configuration_only_loads_models(self):
        for kind in ("dpd", "filter", "mixer", "pa"):
            with self.subTest(kind=kind):
                dialog = StageDialog(Stage(kind, kind, kind, dict(cutoff=1., gain_db=6.)), self.workspace)
                self.assertFalse(dialog.findChildren(QDoubleSpinBox))
                self.assertTrue(dialog.details.isReadOnly())
                self.assertCountEqual([b.text() for b in dialog.findChildren(QPushButton)],
                                      ["从模型库加载…", "导入模型文件…", "应用", "取消"])
                self.assertIn("理想直通", dialog.model_label.text())
                dialog.enabled.setChecked(False)
                dialog.accept()
                self.assertEqual(dialog.stage.settings, {})
                self.assertFalse(dialog.stage.enabled)
                dialog.deleteLater()

    def test_dpd_numeric_controls_keep_editable_fields_without_arrows(self):
        panel = PADPDWidget(self.workspace)
        for field in (panel.memory, panel.degree, panel.cross):
            self.assertEqual(field.buttonSymbols(), QAbstractSpinBox.ButtonSymbols.NoButtons)
            self.assertFalse(field.findChildren(QToolButton))
        edit = panel.memory.lineEdit()
        edit.selectAll()
        QTest.keyClicks(edit, "8")
        panel.memory.interpretText()
        self.assertEqual(panel.memory.value(), 8)

    def test_async_run_selection_and_model_edit(self):
        w = self.workspace
        w.run_chain()
        self.wait_for_worker()
        self.assertTrue(w.export_button.isEnabled())
        w.fit_canvas()
        self.assertEqual(len(w.outputs), 6, "Selecting/repositioning nodes must preserve results")
        pa_node = next(n for n in w.graph.all_nodes() if n.KIND == "pa")
        w.select_node(pa_node)
        self.assertEqual(w.point.currentData(), pa_node.id)
        stage = w.stage_from_node(pa_node)
        model = from_pa(PAModel(np.array([[2.]]), "memPoly", 1, sample_rate=100e6), "两倍增益")
        stage.model = model
        # Exercise the same accepted-dialog route as double-click + library selection.
        class AppliedDialog(QDialog):
            def __init__(self, *args):
                super().__init__()
                self.stage = stage

            def exec(self):
                return QDialog.DialogCode.Accepted

        with patch("link_sim.ui.waveform_workspace.StageDialog", AppliedDialog):
            w.configure_node(pa_node)
        self.assertFalse(w.outputs)
        self.assertFalse(w.export_button.isEnabled())
        w.run_chain()
        self.wait_for_worker()
        mixer = next(n for n in w.graph.all_nodes() if n.KIND == "mixer")
        np.testing.assert_allclose(w.outputs[pa_node.id].x, 2*w.outputs[mixer.id].x)

    def test_save_open_scheme_rebuilds_connections(self):
        w = self.workspace
        dpd = next(n for n in w.graph.all_nodes() if n.KIND == "dpd")
        dpd.set_disabled(True)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"transmitter.json")
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getSaveFileName", return_value=(path, "")):
                w.save_scheme()
            w.graph.delete_nodes(w.graph.all_nodes())
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getOpenFileName", return_value=(path, "")):
                w.open_scheme()
            stages, edges = w.snapshot()
            self.assertEqual(len(stages), 6)
            self.assertEqual(len(edges), 5)
            dpd = next(s for s in stages if s.kind == "dpd")
            dac = next(s for s in stages if s.kind == "dac")
            self.assertFalse(dpd.enabled)
            self.assertIn((dpd.key, dac.key), edges)
            w.run_chain()
            self.wait_for_worker()
            self.assertIn("已旁路", w.outputs[dpd.key].notes[0])


if __name__ == "__main__":
    unittest.main()
