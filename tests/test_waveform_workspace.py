"""Qt integration: actual worker, selection and embedded model configuration."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog

from link_sim.algorithm.device_models import from_pa
from link_sim.algorithm.pa_simulation_core import PAModel
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
        self.assertEqual(len(self.workspace.outputs), 5)

    def test_async_run_selection_and_model_edit(self):
        w = self.workspace
        w.run_chain()
        self.wait_for_worker()
        self.assertTrue(w.export_button.isEnabled())
        w.fit_canvas()
        self.assertEqual(len(w.outputs), 5, "Selecting/repositioning nodes must preserve results")
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
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"transmitter.json")
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getSaveFileName", return_value=(path, "")):
                w.save_scheme()
            w.graph.delete_nodes(w.graph.all_nodes())
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getOpenFileName", return_value=(path, "")):
                w.open_scheme()
            stages, edges = w.snapshot()
            self.assertEqual(len(stages), 5)
            self.assertEqual(len(edges), 4)
            w.run_chain()
            self.wait_for_worker()


if __name__ == "__main__":
    unittest.main()
