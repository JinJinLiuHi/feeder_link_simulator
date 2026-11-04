"""Numerical and portable-scheme regressions for the transmitter chain."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import signal

from link_sim.algorithm.device_models import (
    FILTER_DEFAULTS, DeviceModel, from_pa, read_model_file, write_model_file,
)
from link_sim.algorithm.model_store import ModelStore
from link_sim.algorithm.mixer_noise_core import phase_noise_psd
from link_sim.algorithm.pa_dpd_core import signature
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
        taps = np.array([.25, .5j, -.25])
        model = DeviceModel("filter", dict(fs=self.fs, impl="外部系数", band="低通"),
                            dict(b=taps, a=np.array([2.])), "复数三抽头")
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            loaded = store.get(store.save(model))
        np.testing.assert_array_equal(loaded.arrays["h"], taps / 2)
        # Include the convolution tail so the FFT identity holds across every bin.
        x = np.pad(self.wave.x, (0, len(taps)-1))
        source = Stage("s", "source", "基带", dict(fs=self.fs, iq=np.column_stack((x.real, x.imag))))
        filt = Stage("f", "filter", "滤波器", dict(cutoff=1.), loaded)
        y = run_graph([source, filt], [("s", "f")])["f"]
        h = np.fft.fft(taps / 2, n=len(x))
        assert_allclose(np.fft.fft(y.x), h*np.fft.fft(x), atol=1e-12)
        self.assertEqual(y.start, 2)

    def test_iir_impulse_is_portable_without_truncating_the_runtime_response(self):
        pole = .9999
        model = DeviceModel("filter", dict(fs=self.fs, impl="外部系数", band="低通"),
                            dict(b=[1.], a=[1., -pole]), "慢衰减 IIR")
        filt = Stage("f", "filter", "滤波器", model=model)
        x = np.zeros(8192, dtype=complex)
        x[0] = 1. + .5j
        source = Stage("s", "source", "基带", dict(fs=self.fs, iq=np.column_stack((x.real, x.imag)).tolist()))
        edges = [("s", "f")]
        with tempfile.TemporaryDirectory() as folder:
            for suffix in ("json", "npz"):
                path = Path(folder)/f"filter.{suffix}"
                write_model_file(path, model)
                loaded = read_model_file(path, "filter")
                self.assertEqual(loaded.parameters["fs"], self.fs)
                assert_allclose(loaded.arrays["h"], pole**np.arange(4096), rtol=1e-12)
            path = Path(folder)/"scheme.json"
            save_chain(path, [source, filt], edges, {})
            stages, edges, _, backend = load_chain(path)
        np.testing.assert_array_equal(stages[1].model.arrays["h"], loaded.arrays["h"])
        y = run_graph(stages, edges, backend)["f"].x
        assert_allclose(y, x[0]*pole**np.arange(len(x)), rtol=1e-12)

    def test_import_complex_impulse_response_and_convolve_in_time(self):
        h = np.array([.6, .2j, -.1])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"impulse.npz"
            np.savez(path, h=h, fs=self.fs)
            model = read_model_file(path, "filter")
        y = process_stage(Stage("f", "filter", "滤波器", model=model), self.wave)
        x = self.wave.x
        expected = h[0]*x + h[1]*np.pad(x[:-1], (1, 0)) + h[2]*np.pad(x[:-2], (2, 0))
        assert_allclose(y.x, expected, atol=1e-15)

    def test_iir_and_rrc_preserve_signed_tone_amplitude_and_phase(self):
        frequencies = np.array([12.5e6, -18.75e6])
        z = np.exp(-2j*np.pi*frequencies/self.fs)
        tones = np.exp(2j*np.pi*np.arange(4096)[:, None]*frequencies/self.fs)
        tones *= [.7, .3j]
        incoming = Wave(tones.sum(axis=1), self.fs)
        p = dict(fs=self.fs, impl="外部系数", band="低通")
        b, a = np.array([.3, .1]), np.array([2., -.7, .1])
        sos = np.array([[.3, .1, 0., 2., -.7, .1]])
        h = np.array([.2, .6, .2])
        for params, arrays, numerator, denominator in (
            (p, dict(b=b, a=a), b, a),
            (p, dict(b=b, a=a, sos=sos), b, a),
            (dict(FILTER_DEFAULTS, fs=self.fs, impl="RRC 成形"), dict(h=h), h, [1.]),
        ):
            with self.subTest(coefficients=tuple(arrays)):
                model = DeviceModel("filter", params, arrays)
                y = process_stage(Stage("f", "filter", "滤波器", model=model), incoming)
                response = np.polynomial.polynomial.polyval(z, numerator) / np.polynomial.polynomial.polyval(z, denominator)
                assert_allclose(y.x[128:], (tones @ response)[128:], atol=1e-12)

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

    def test_mixer_uses_saved_phase_noise_and_input_carrier_metadata(self):
        incoming = Wave(self.wave.x[:-1], self.fs, carrier_hz=12e9)
        model = DeviceModel("mixer", dict(fs=10e6, sim_time=.001, f_if_in=1e6,
            anchors=[[1e3, -60.], [1e6, -90.]]))
        stage = Stage("m", "mixer", "上变频器", dict(carrier_hz=14e9, gain_db=6.), model)
        noisy = process_stage(stage, incoming)
        self.assertGreater(np.max(abs(noisy.x-incoming.x)), .001)
        self.assertEqual(noisy.carrier_hz, incoming.carrier_hz)
        self.assertEqual(noisy.fs, incoming.fs)
        assert_allclose(abs(noisy.x), abs(incoming.x), rtol=1e-12)

    def test_mixer_default_example_reproduces_the_interpolated_psd_and_phase(self):
        fs, count = 10e6, 10000
        anchors = [[1e3, -60.], [1e4, -80.], [1e5, -110.], [1e6, -130.]]
        offsets, levels, psd = phase_noise_psd(anchors, fs, count)
        self.assertEqual(offsets[2], 2000.)
        self.assertAlmostEqual(levels[2], -66.0205999133, places=9)
        assert_allclose(psd[[1, 2, 10, 100, 1000, -1]], [1e-6, 2.5e-7, 1e-8, 1e-11, 1e-13, 1e-13])
        model = DeviceModel("mixer", dict(fs=fs, sim_time=.001, f_if_in=1e6, anchors=anchors))
        incoming = Wave(np.full(count, .1+.1j), fs)
        y = process_stage(Stage("m", "mixer", "上变频器", model=model), incoming)
        phi = np.angle(y.x / incoming.x)
        self.assertAlmostEqual(phi.mean(), 0., places=14)
        self.assertAlmostEqual(phi[0], -.0574762298984, places=11)
        self.assertAlmostEqual(phi.std(), .0603864242284, places=11)

    def test_saved_pa_is_nonlinear_and_not_renormalized(self):
        pa = PAModel(np.array([[2., 0., -.5]]), "memPoly", 3, sample_rate=self.fs)
        model = from_pa(pa, "测试功放")
        y = process_stage(Stage("p", "pa", "功放", {}, model), self.wave)
        assert_allclose(y.x, self.wave.x*(2-.5*abs(self.wave.x)**2))

    def test_dpd_pair_is_saved_and_checked_with_independent_bypass(self):
        coefficients = np.array([[2.], [.2]], dtype=complex)
        coefficients.imag[0, 0] = -0.
        pa = PAModel(coefficients, "memPoly", 1, sample_rate=self.fs)
        dpd = PAModel(np.array([[.5], [.1]]), "memPoly", 1, sample_rate=self.fs,
                      fit_info=dict(gainReal=1., gainImag=0., paSignature=signature(pa)), role="predistorter")
        pa_model, dpd_model = from_pa(pa, "配对功放"), from_pa(dpd, "测试 DPD")
        edges = [("s", "d"), ("d", "a"), ("a", "p")]
        with tempfile.TemporaryDirectory() as folder:
            store = ModelStore(Path(folder)/"models.sqlite3")
            with self.assertRaisesRegex(ValueError, "配对"):
                store.save(dpd_model)
            self.assertEqual(store.list(), [])
            dpd_id = store.save(dpd_model, paired_pa=pa_model)
            self.assertEqual(store.paired_power_amplifier(dpd_id).name, pa_model.name)
            self.assertEqual(len(store.list("pa")), 1)
            model_file = Path(folder)/"pa.json"
            write_model_file(model_file, pa_model)
            pa_model = read_model_file(model_file, "pa")
            with store.connection() as db:
                self.assertIsNotNone(db.execute("SELECT pa_id FROM dpd_pa_pairs WHERE dpd_id=?", (dpd_id,)).fetchone())
                db.execute("DELETE FROM dpd_pa_pairs")  # Simulate a legacy record.
            self.assertEqual(store.paired_power_amplifier(dpd_id).name, pa_model.name)
            stages = [Stage("s", "source", "信号源"), Stage("d", "dpd", "DPD", model=store.get(dpd_id)),
                      Stage("a", "dac", "AD/DA"), Stage("p", "pa", "功放", model=pa_model)]
            scheme = Path(folder)/"dpd.json"
            save_chain(scheme, stages, edges, {})
            stages, edges, _, _ = load_chain(scheme)
        source, predistorter, _, amplifier = stages
        out = run_graph(stages, edges)
        assert_allclose(out["d"].x, dpd.predict(out["s"].x))
        assert_allclose(out["p"].x, pa.predict(out["d"].x))
        self.assertEqual(out["p"].start, 2)
        amplifier.model = from_pa(PAModel(np.array([[3.]]), "memPoly", 1, sample_rate=self.fs))
        with self.assertRaisesRegex(ValueError, "不配对"):
            run_graph(stages, edges)
        amplifier.enabled = False
        out = run_graph(stages, edges)
        assert_allclose(out["p"].x, out["d"].x, atol=0, rtol=0)
        self.assertEqual(out["p"].start, out["d"].start)
        amplifier.enabled, predistorter.enabled = True, False
        out = run_graph(stages, edges)
        assert_allclose(out["d"].x, out["s"].x, atol=0, rtol=0)
        assert_allclose(out["p"].x, 3*out["s"].x)
        source.enabled = False
        self.assertFalse(np.any(run_graph(stages, edges)["s"].x))

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
        for kind in kinds[1:]:
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(out[kind].x, out["source"].x)
                self.assertEqual(out[kind].start, out["source"].start)
                self.assertEqual(out[kind].carrier_hz, out["source"].carrier_hz)
                self.assertIn("理想直通", out[kind].notes[0])
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
