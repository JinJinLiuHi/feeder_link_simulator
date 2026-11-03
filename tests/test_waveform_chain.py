"""Numerical and portable-scheme regressions for the transmitter chain."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import signal

from link_sim.algorithm.device_models import DeviceModel, from_pa
from link_sim.algorithm.model_store import ModelStore
from link_sim.algorithm.pa_simulation_core import PAModel, load_waveform
from link_sim.algorithm.waveform_chain import (
    Wave, Stage, default_settings, export_wave, import_source, load_chain,
    order_graph, process_stage, run_graph, save_chain, source_wave,
)
from link_sim.algorithm.waveform_gnuradio import native_block, runtime_status


class WaveformChainTests(unittest.TestCase):
    def setUp(self):
        self.fs = 100e6
        self.wave = source_wave(default_settings("source"))

    def test_seeded_source_rms_and_repeatability(self):
        other = source_wave(default_settings("source"))
        assert_allclose(self.wave.x, other.x, atol=0)
        self.assertAlmostEqual(np.sqrt(np.mean(abs(self.wave.x)**2)), .2)
        for mode in ("BPSK", "16QAM"):
            self.assertTrue(np.all(np.isfinite(source_wave(dict(default_settings("source"), modulation=mode)).x)))

    def test_adda_without_saved_model_is_ideal_passthrough(self):
        x = Wave(np.array([-.9-1.3j, -.3+.3j, 0., .9+1.3j]), self.fs)
        stage = Stage("d", "dac", "DAC", dict(bits=2, noise_density=-400.))
        y = process_stage(stage, x)
        assert_allclose(y.x, x.x, atol=1e-15)
        self.assertIn("理想直通", y.notes[0])

    def test_library_filter_uses_saved_coefficients(self):
        model = DeviceModel("filter", dict(fs=self.fs, impl="外部系数", band="低通"),
                            dict(b=np.array([.25, .5, .25]), a=np.array([1.])), "三抽头")
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            loaded = store.get(store.save(model))
        stage = Stage("f", "filter", "滤波器", dict(cutoff=1.), loaded)
        y = process_stage(stage, self.wave)
        assert_allclose(y.x, signal.lfilter(model.arrays["b"], [1.], self.wave.x))
        self.assertEqual(y.start, 2)

    def test_saved_adda_error_comes_from_original_model_output(self):
        p = dict(n_bits=8, fs=self.fs, nfft=1024, f_sig=1e6, noise_density_dbfs_hz=-400.,
                 sfdr_target=60., device_profile="自定义", spur_freqs=[2e6], spur_dbfs=[-60.])
        model = DeviceModel("adda", p).validated()
        y = process_stage(Stage("d", "dac", "DAC", dict(bits=2), model), self.wave)
        from link_sim.algorithm.adc_noise_core import calibrate_model_noise, generate_model_noise
        reference = calibrate_model_noise(model)
        noise, _ = generate_model_noise(reference, len(self.wave.x), self.fs)
        assert_allclose(y.x, self.wave.x+noise)
        self.assertGreater(np.max(abs(y.x-self.wave.x)), 1e-4)

    def test_mixer_carrier_metadata_and_saved_phase_noise(self):
        stage = Stage("m", "mixer", "上变频器", dict(carrier_hz=14e9, gain_db=6.))
        ideal = process_stage(stage, self.wave)
        assert_allclose(ideal.x, self.wave.x*10**(.3))
        self.assertEqual(ideal.carrier_hz, 14e9)
        stage.model = DeviceModel("mixer", dict(fs=self.fs, sim_time=.0001, f_if_in=10e6,
            anchors=[[1e3, -60.], [1e6, -90.]])).validated()
        noisy = process_stage(stage, self.wave)
        self.assertGreater(np.max(abs(noisy.x-ideal.x)), .001)
        assert_allclose(abs(noisy.x), abs(ideal.x), rtol=1e-12)

    def test_saved_pa_is_nonlinear_and_not_renormalized(self):
        pa = PAModel(np.array([[2., 0., -.5]]), "memPoly", 3, sample_rate=self.fs)
        model = from_pa(pa, "测试功放")
        y = process_stage(Stage("p", "pa", "功放", {}, model), self.wave)
        assert_allclose(y.x, self.wave.x*(2-.5*abs(self.wave.x)**2))

    def test_pa_valid_region_propagates_upstream_edges(self):
        # Use a GMP with lookahead for the boundary regression.
        pa = PAModel(np.array([[1., .1, 0., 0.], [0., 0., 0., 0.]]),
                     "gmp", 2, cross_depth=1, sample_rate=self.fs)
        w = Wave(self.wave.x, self.fs, start=5, stop=len(self.wave.x)-7)
        y = process_stage(Stage("p", "pa", "功放", {}, from_pa(pa)), w)
        self.assertEqual(y.start, 7)
        self.assertEqual(y.stop, len(self.wave.x)-8)

    def test_sample_rate_mismatch_rejected(self):
        model = from_pa(PAModel(np.array([[1.]]), "memPoly", 1, sample_rate=200e6))
        with self.assertRaisesRegex(ValueError, "采样率"):
            process_stage(Stage("p", "pa", "功放", {}, model), self.wave)

    def test_frequency_shift_and_awgn_snr(self):
        n = 16384
        w = Wave(np.ones(n, dtype=complex), self.fs)
        y = process_stage(Stage("f", "frequency", "频偏", dict(offset_hz=self.fs*128/n)), w)
        self.assertEqual(np.argmax(abs(np.fft.fft(y.x))), 128)
        y = process_stage(Stage("a", "awgn", "噪声", dict(snr_db=20., seed=7)), w)
        measured = 10*np.log10(np.mean(abs(w.x)**2)/np.mean(abs(y.x-w.x)**2))
        self.assertAlmostEqual(measured, 20., delta=.15)

    def test_reconnected_graph_and_branches_change_execution(self):
        s = Stage("s", "source", "信号源", default_settings("source"))
        a = Stage("a", "gain", "增益A", dict(gain_db=6.))
        b = Stage("b", "gain", "增益B", dict(gain_db=-6.))
        out = run_graph([b, s, a], [("s", "a"), ("a", "b")])
        assert_allclose(out["b"].x, out["s"].x)
        out = run_graph([b, s, a], [("s", "a"), ("s", "b")])
        assert_allclose(out["b"].x, out["s"].x*10**(-.3))

    def test_bad_topologies_are_rejected(self):
        s = Stage("s", "source", "信号源", default_settings("source"))
        a = Stage("a", "gain", "增益", {})
        for stages, edges in (([s, a], []), ([s, a], [("a", "a")]),
                              ([s, a], [("s", "a"), ("s", "a")]),
                              ([s, s], []), ([a], [])):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                order_graph(stages, edges)

    def test_portable_scheme_embeds_iq_and_model_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            input_path = folder/"input.npz"
            export_wave(input_path, self.wave)
            source = Stage("s", "source", "外部信号", import_source(input_path, 1.))
            model = from_pa(PAModel(np.array([[2+1j]]), "memPoly", 1, sample_rate=self.fs), "模型快照")
            pa = Stage("p", "pa", "功放", {}, model)
            stages, edges = [source, pa], [("s", "p")]
            expected = run_graph(stages, edges)
            path = folder/"test.chain.json"
            save_chain(path, stages, edges, {"s": [0, 0], "p": [200, 0]})
            input_path.unlink()
            loaded, links, positions, backend = load_chain(path)
            actual = run_graph(loaded, links, backend)
            assert_allclose(actual["p"].x, expected["p"].x)
            self.assertEqual(positions["p"], [200, 0])
            export_wave(folder/"output.npz", actual["p"])
            x, fs = load_waveform(folder/"output.npz")
            assert_allclose(x, actual["p"].x)
            self.assertEqual(fs, self.fs)

    def test_default_transmitter_and_cancellation(self):
        kinds = ("source", "dac", "filter", "mixer", "pa")
        stages = [Stage(k, k, k, default_settings(k)) for k in kinds]
        edges = list(zip(kinds, kinds[1:]))
        out = run_graph(stages, edges)
        self.assertEqual(len(out), 5)
        self.assertEqual(out["pa"].start, 64)
        self.assertEqual(out["pa"].carrier_hz, 14e9)
        with self.assertRaises(InterruptedError):
            run_graph(stages, edges, cancelled=lambda: True)

    @unittest.skipUnless(runtime_status()[0], "GNU Radio runtime not installed")
    def test_gnuradio_native_blocks_match_numerical_reference(self):
        assert_allclose(native_block(self.wave.x, "gain", gain=2.), 2*self.wave.x, atol=1e-6)
        taps = signal.firwin(17, .2)
        assert_allclose(native_block(self.wave.x, "fir", taps=taps), signal.lfilter(taps, [1.], self.wave.x), atol=1e-6)
        expected = self.wave.x*np.exp(2j*np.pi*1e6*np.arange(len(self.wave.x))/self.fs)
        assert_allclose(native_block(self.wave.x, "frequency", fs=self.fs, offset=1e6), expected, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
