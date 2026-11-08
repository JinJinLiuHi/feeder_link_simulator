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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QPushButton, QStyle, QStyleOptionSpinBox, QToolButton

from link_sim.algorithm.device_models import from_pa
from link_sim.algorithm.pa_dpd import PADPDWidget
from link_sim.algorithm.pa_simulation_core import PAModel
from link_sim.algorithm.waveform_chain import Stage, Wave
from link_sim.algorithm.waveform_chain import default_settings
from link_sim.ui.waveform_workspace import StageDialog, WaveformWorkspace
from link_sim.ui.workspace import EMBEDDED_STYLE, LazyDialogPage
from waveform_fixtures import configure_workspace_rrc, rrc_model


class WaveformWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.workspace = WaveformWorkspace()
        configure_workspace_rrc(self.workspace)

    def tearDown(self):
        self.workspace.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    def wait_for_worker(self, expected_outputs=12):
        deadline = time.monotonic()+15
        while self.workspace.worker and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(self.workspace.worker, "Background computation did not finish")
        self.assertEqual(len(self.workspace.outputs), expected_outputs)

    def test_preview_and_export_source_rrc_without_valid_downstream_modules(self):
        w = self.workspace
        source = next(n for n in w.graph.all_nodes() if n.KIND == "frame_source")
        rrc = next(n for n in w.graph.all_nodes() if n.KIND == "rrc_tx")
        pa = next(n for n in w.graph.all_nodes() if n.KIND == "pa")
        # A broken stored model and disconnected receiver must not be inspected.
        pa.set_property("device_model", {"kind": "pa"}, push_undo=False)
        w.graph.delete_nodes([n for n in w.graph.all_nodes() if n.KIND == "sync"])
        self.assertGreaterEqual(w.point.findData(source.id), 0)
        self.assertGreaterEqual(w.point.findData(rrc.id), 0)
        w.select_node(rrc)
        with patch("link_sim.ui.waveform_workspace.QMessageBox.warning") as warning:
            w.preview_button.click()
            self.wait_for_worker(2)
            warning.assert_not_called()
        self.assertEqual(set(w.outputs), {source.id, rrc.id})
        self.assertEqual(w.point.currentData(), rrc.id)
        with tempfile.TemporaryDirectory() as folder:
            for node, fs in ((source, 700e6), (rrc, 2.8e9)):
                w.select_node(node)
                self.assertTrue(w.export_button.isEnabled())
                self.assertEqual(len(w.figure.axes), 3)
                path = str(Path(folder)/(node.KIND+".npz"))
                with patch("link_sim.ui.waveform_workspace.QFileDialog.getSaveFileName", return_value=(path, "")):
                    w.export_selected()
                with np.load(path) as data:
                    np.testing.assert_array_equal(data["x"], w.outputs[node.id].x)
                    self.assertEqual(float(data["sampleRate"]), fs)
                    self.assertEqual(len(data["frequencyHz"]), len(w.outputs[node.id].valid))
                    self.assertEqual(len(data["spectrumDb"]), len(data["frequencyHz"]))
                    self.assertNotIn("carrierHz", data)
        w.select_node(pa)
        self.assertIsNone(w.observed_wave())
        self.assertFalse(w.export_button.isEnabled())
        self.assertFalse(w.figure.axes, "Do not show the previous module's signal for an uncomputed output")

    def test_spectrum_view_uses_baseband_frequency_and_valid_samples(self):
        w = self.workspace
        node = next(n for n in w.graph.all_nodes() if n.KIND == "frame_source")
        x = np.zeros(4096, dtype=complex)
        x[128:2176] = .25*np.exp(2j*np.pi*.125*np.arange(2048))
        w.outputs[node.id] = Wave(x, 100e6, carrier_hz=12e9, start=128, stop=2176)
        w.refresh_observation_points(node.id)
        w.plot_view.setCurrentIndex(w.plot_view.findData("spectrum"))
        self.assertEqual(len(w.figure.axes), 1)
        ax = w.figure.axes[0]
        frequencies, levels = ax.lines[0].get_data()
        self.assertEqual(frequencies[np.argmax(levels)], 12.5)
        self.assertAlmostEqual(max(levels), 20*np.log10(.25), places=10)
        self.assertNotIn("载频", w.summary.text())
        self.assertIn("基带频率", ax.get_xlabel())
        w.plot_view.setCurrentIndex(w.plot_view.findData("all"))
        self.assertEqual(len(w.figure.axes), 3)

    def test_device_configuration_only_loads_models(self):
        for kind in ("dpd", "filter", "mixer", "pa", "rrc_tx", "rrc_rx"):
            with self.subTest(kind=kind):
                dialog = StageDialog(Stage(kind, kind, kind, dict(cutoff=1., gain_db=6.)), self.workspace)
                self.assertFalse(dialog.findChildren(QDoubleSpinBox))
                self.assertTrue(dialog.details.isReadOnly())
                self.assertCountEqual([b.text() for b in dialog.findChildren(QPushButton)],
                                      ["从模型库加载…", "导入模型文件…", "应用", "取消"])
                self.assertIn("理想直通" if kind not in ("rrc_tx", "rrc_rx") else "RRC", dialog.model_label.text())
                dialog.enabled.setChecked(False)
                dialog.accept()
                self.assertEqual(dialog.stage.settings, {})
                self.assertFalse(dialog.stage.enabled)
                dialog.deleteLater()

    def test_rrc_load_and_import_use_model_parameters_without_input_controls(self):
        from link_sim.algorithm.device_models import write_model_file
        dialog = StageDialog(Stage("r", "rrc_tx", "发射 RRC"), self.workspace)
        first, second = rrc_model(4), rrc_model(8, .35, 10)
        class Library(QDialog):
            selected_model = first
            def __init__(self, *args):
                super().__init__()
            def exec(self):
                return QDialog.DialogCode.Accepted
        with patch("link_sim.ui.waveform_workspace.ModelLibraryDialog", Library):
            dialog.choose_model()
        self.assertEqual(dialog.stage.model.parameters["sps"], 4)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/"rrc.json")
            write_model_file(path, second)
            with patch("link_sim.ui.waveform_workspace.QFileDialog.getOpenFileName", return_value=(path, "")):
                dialog.import_model()
        self.assertFalse(dialog.findChildren(QDoubleSpinBox))
        self.assertEqual(dialog.stage.model.parameters["sps"], 8)
        np.testing.assert_array_equal(dialog.stage.model.arrays["h"], second.arrays["h"])
        dialog.accept()
        self.assertEqual(dialog.stage.settings, {})
        dialog.deleteLater()

    def test_frame_source_editing_independent_modcod_and_live_bit_count(self):
        from link_budget.algorithm.ber_core.catalog import apsk_definition, session_catalog, update_session_catalog
        saved = session_catalog()
        custom = apsk_definition(16, [4, 12], [1., 2.85], [0., 15.])
        code = dict(id="test_conv", name="已新增卷积码", family="conv", k=128, n=260,
                    length=3, polys=["7", "5"], terminate=True)
        update_session_catalog(saved["modulations"]+[custom], saved["codes"]+[code])
        try:
            dialog = StageDialog(Stage("source", "frame_source", "发射信号", default_settings("frame_source")), self.workspace)
            self.assertEqual(dialog.fields["symbol_rate"][0].value(), 700.)
            self.assertTrue({"sps", "rolloff", "span", "k", "n"}.isdisjoint(dialog.fields))
            dialog.fields["segments"][0].setValue(1)
            dialog.fields["words"][0].setValue(3)
            editor = dialog.frame_editor
            self.assertEqual(editor.table.rowCount(), 3)
            for row, name in enumerate(("BPSK", "QPSK", "16QAM")):
                editor.combos[row][0].setCurrentText(name)
            self.assertIn("1,536 bit", editor.summary.text())
            self.assertIn("2,880", editor.summary.text())
            mod, family, coding = editor.combos[2]
            self.assertGreaterEqual(mod.findText(custom["name"]), 0)
            family.setCurrentIndex(family.findData("conv"))
            self.assertEqual(coding.currentData()["id"], "test_conv")
            self.assertIn("1,152 bit", editor.summary.text())
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(dialog.stage.settings["business_frames"][0][2]["code"]["id"], "test_conv")
            self.assertEqual(dialog.stage.settings["business_frames"][0][0]["mod"]["name"], "BPSK")
            self.assertNotIn("sps", dialog.stage.settings)
            dialog.deleteLater()
        finally:
            update_session_catalog(saved["modulations"], saved["codes"])

    def test_dpd_numeric_controls_keep_editable_fields_without_arrows(self):
        panel = PADPDWidget(self.workspace)
        panel.setStyleSheet(EMBEDDED_STYLE)
        for field in (panel.memory, panel.degree, panel.cross):
            self.assertFalse(field.findChildren(QToolButton))
            option = QStyleOptionSpinBox()
            field.initStyleOption(option)
            for control in (QStyle.SubControl.SC_SpinBoxUp, QStyle.SubControl.SC_SpinBoxDown):
                self.assertEqual(field.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, field).width(), 0)
        edit = panel.memory.lineEdit()
        edit.selectAll()
        QTest.keyClicks(edit, "8")
        panel.memory.interpretText()
        self.assertEqual(panel.memory.value(), 8)

    def test_embedded_pa_fields_have_a_mouse_editable_text_area(self):
        page = LazyDialogPage("link_sim.algorithm.pa_model", "PAModelDialog")
        page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        page.resize(1320, 900)
        page.show()
        self.app.processEvents()
        try:
            dialog = page.dialog
            for field, value in ((dialog.e_memlen, "8"), (dialog.e_deglen, "7")):
                self.assertGreater(field.lineEdit().width(), field.width() / 2)
                QTest.mouseClick(field.lineEdit(), Qt.MouseButton.LeftButton, pos=field.lineEdit().rect().center())
                QTest.keyClick(field.lineEdit(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
                QTest.keyClicks(field.lineEdit(), value)
                QTest.keyClick(field.lineEdit(), Qt.Key.Key_Return)
                self.assertEqual(field.value(), int(value))
            dialog.combo_model.setCurrentIndex(1)
            self.assertTrue(dialog.e_crossdepth.isEnabled())
        finally:
            page.close()
            page.deleteLater()

    def test_zoomed_dual_graph_keeps_all_titles_and_lane_labels(self):
        w = self.workspace
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        w.resize(1300, 850)
        w.show()
        w.link_mode.setCurrentIndex(1)
        self.app.processEvents()
        w.fit_canvas()
        w.graph.widget.grab()
        self.assertEqual(len(w.graph.all_nodes()), 26)
        for node in w.graph.all_nodes():
            self.assertTrue(node.view.text_item.isVisible(), node.name())
            if node.KIND not in ("xpic", "fastica"):
                self.assertRegex(node.name(), r"^[AB] · ")

    def test_async_run_selection_and_model_edit(self):
        w = self.workspace
        self.assertFalse(hasattr(w, "backend"))
        w.run_chain()
        self.wait_for_worker()
        self.assertTrue(w.export_button.isEnabled())
        w.fit_canvas()
        self.assertEqual(len(w.outputs), 12, "Selecting/repositioning nodes must preserve results")
        pa_node = next(n for n in w.graph.all_nodes() if n.KIND == "pa")
        w.select_node(pa_node)
        self.assertEqual(w.point.currentData(), pa_node.id)
        stage = w.stage_from_node(pa_node)
        model = from_pa(PAModel(np.array([[2.]]), "memPoly", 1,
                               sample_rate=w.outputs[pa_node.id].fs), "两倍增益")
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
            self.assertEqual(len(stages), 12)
            self.assertEqual(len(edges), 11)
            dpd = next(s for s in stages if s.kind == "dpd")
            dac = next(s for s in stages if s.kind == "dac")
            self.assertFalse(dpd.enabled)
            self.assertIn((dpd.key, dac.key), edges)
            w.run_chain()
            self.wait_for_worker()
            self.assertIn("已旁路", w.outputs[dpd.key].notes[0])


if __name__ == "__main__":
    unittest.main()
