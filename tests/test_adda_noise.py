"""Original estimated ADC noise floor -> density -> complex additive noise."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose
from PySide6 import QtCore  # Qt DLLs precede Matplotlib, as in main.py.

from link_sim.algorithm.ad_model import run_adc_model
from link_sim.algorithm.adc_noise_core import ADCNoiseEstimate, calibrate_model_noise, generate_model_noise
from link_sim.algorithm.device_models import DeviceModel
from link_sim.algorithm.waveform_chain import Stage, Wave, process_stage, run_graph, save_chain, load_chain, export_wave


def device(**parameters):
    p = dict(n_bits=12, fs=64e6, nfft=8192, f_sig=8e6, noise_density_dbfs_hz=-150.,
             sfdr_target=60., device_profile="测试原模型", spur_freqs=[12e6], spur_dbfs=[-60.])
    p.update(parameters)
    return DeviceModel("adda", p, name="原 ADC 模型").validated()


class ADDANoiseFloorTests(unittest.TestCase):
    def test_estimate_matches_metric_shown_by_original_editor(self):
        model = device()
        p = model.parameters.copy()
        keys = ("n_bits", "fs", "nfft", "f_sig", "spur_freqs", "spur_dbfs", "noise_density_dbfs_hz")
        result = run_adc_model(**{key: p[key] for key in keys})
        estimate = calibrate_model_noise(model)
        self.assertEqual(estimate.floor_dbfs, result["metrics"]["noise_floor_dbfs"])
        self.assertEqual(estimate.bin_width_hz, p["fs"]/p["nfft"])
        self.assertEqual(model.parameters, p)

    def test_displayed_minus_105_06_dbfs_example(self):
        estimate = ADCNoiseEstimate(-105.06, 3e9/65536)
        self.assertAlmostEqual(estimate.density_dbfs_hz, -151.66641324095963, places=10)
        self.assertAlmostEqual(estimate.noise_power(100e6)/6.81331825943935e-8, 1., places=12)
        self.assertAlmostEqual(np.sqrt(estimate.noise_power(100e6)), .00026102333726008773, places=15)

    def test_gaussian_power_iq_split_and_absence_of_spur_lines(self):
        estimate = calibrate_model_noise(device())
        n, power = generate_model_noise(estimate, 131072, 16e6)
        self.assertAlmostEqual(np.mean(abs(n)**2)/power, 1., delta=.02)
        self.assertAlmostEqual(np.mean(n.real**2)/(power/2), 1., delta=.02)
        self.assertAlmostEqual(np.mean(n.imag**2)/(power/2), 1., delta=.02)
        spectrum = abs(np.fft.fft(n))**2
        self.assertLess(spectrum.max()/spectrum.mean(), 30.)

    def test_original_noise_floor_controls_power_and_no_second_quantizer(self):
        # Isolate noise density from the original model's full-scale sine/spur
        # clipping: the editor itself is intentionally left unchanged.
        quiet = calibrate_model_noise(device(n_bits=24, noise_density_dbfs_hz=-150., sfdr_target=120., spur_dbfs=[-120.]))
        loud = calibrate_model_noise(device(n_bits=24, noise_density_dbfs_hz=-130., sfdr_target=120., spur_dbfs=[-120.]))
        self.assertGreater(loud.noise_power(16e6), 50*quiet.noise_power(16e6))
        s = Wave(np.full(8192, 2+3j), 16e6)
        stage = Stage("a", "dac", "AD/DA", {}, device())
        out = process_stage(stage, s)
        assert_allclose(out.x, s.x+out.added_error, rtol=0, atol=0)
        self.assertGreater(out.x.real.min(), 1.9)

    def test_noise_does_not_scale_with_live_signal_and_old_controls_are_ignored(self):
        model = device()
        s = Wave(np.full(8192, .2+.1j), 16e6)
        stage = Stage("a", "dac", "AD/DA", {}, model)
        out = process_stage(stage, s)
        weak = process_stage(stage, Wave(s.x/100, s.fs))
        zero = process_stage(stage, Wave(np.zeros_like(s.x), s.fs))
        assert_allclose(out.added_error, weak.added_error, rtol=0, atol=0)
        assert_allclose(zero.x, out.added_error, rtol=0, atol=0)
        old = Stage("a", "dac", "AD/DA", dict(seed=99, noise_offset_hz=1e9, noise_bandwidth_hz=1e12), model)
        assert_allclose(process_stage(old, s).x, out.x, rtol=0, atol=0)

    def test_floor_is_density_not_total_power_and_fft_size_does_not_change_density(self):
        first = ADCNoiseEstimate(-105.06, 3e9/65536)
        # Doubling FFT length halves bin width and lowers the per-bin floor by 3.01 dB.
        second = ADCNoiseEstimate(-105.06-10*np.log10(2), 3e9/131072)
        self.assertAlmostEqual(first.noise_power(100e6)/second.noise_power(100e6), 1., places=12)
        self.assertAlmostEqual(first.noise_power(50e6)*2/first.noise_power(100e6), 1., places=12)
        with self.assertRaises(ValueError):
            first.noise_power(0.)

    def test_portable_original_model_snapshot_and_error_export(self):
        model = device()
        stages = [Stage("s", "source", "输入", dict(fs=16e6, count=8192)),
                  Stage("a", "dac", "AD/DA", {}, model), Stage("b", "dac", "AD/DA", {}, model)]
        edges = [("s", "a"), ("a", "b")]
        before = run_graph(stages, edges)
        self.assertFalse(np.array_equal(before["a"].added_error, before["b"].added_error))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"chain.json"
            save_chain(path, stages, edges, {})
            loaded, links, _, backend = load_chain(path)
            after = run_graph(loaded, links, backend)
            assert_allclose(after["b"].x, before["b"].x, rtol=0, atol=0)
            self.assertEqual(loaded[1].model.parameters, model.parameters)
            export_wave(Path(folder)/"output.npz", after["a"])
            with np.load(Path(folder)/"output.npz", allow_pickle=False) as data:
                assert_allclose(data["x"], before["s"].x+data["addedError"])


if __name__ == "__main__":
    unittest.main()
