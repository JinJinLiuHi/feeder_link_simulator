"""Actual Sionna waveform generation, FFT references and the DPD generation path."""

import builtins
import sys
import unittest
from unittest.mock import patch
import numpy as np

from link_sim.algorithm.pa_ofdm import generate_ofdm, parameters


class PAOFDMTests(unittest.TestCase):
    def test_real_sionna_generation_matches_cp_and_frequency_grid(self):
        x, rate, reference = generate_ofdm(5e6, 1, seed=73, return_reference=True)
        size, cp = reference["fftSize"], reference["cyclicPrefixLengths"]
        self.assertEqual(rate, parameters(5e6, 1)["sampleRate"])
        self.assertEqual(len(x), int(np.sum(size+cp)))
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(float(np.max(abs(x))), 1.)
        start = 0
        for index, length in enumerate(cp):
            np.testing.assert_array_equal(x[start:start+length], x[start+size:start+size+length])
            grid = np.fft.fftshift(np.fft.fft(x[start+length:start+length+size], norm="ortho"))
            np.testing.assert_allclose(grid, reference["grid"][index], atol=1e-12)
            start += size+length

    def test_normal_cp_waveform_preserves_variable_prefix_lengths(self):
        x, rate, reference = generate_ofdm(50e6, 1, return_reference=True)
        self.assertEqual(rate, 61.44e6)
        self.assertEqual(len(x), 614400)
        np.testing.assert_array_equal(reference["cyclicPrefixLengths"][:8], [320,288,288,288,288,288,288,320])

    def test_dependency_error_includes_original_exception_and_running_python(self):
        original_import = builtins.__import__
        error = ImportError("cannot import name OFDMModulator")
        def import_backend(name, *args, **kwargs):
            if name == "sionna.phy.ofdm":
                raise error
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=import_backend), self.assertRaises(RuntimeError) as caught:
            generate_ofdm(5e6, 1)
        self.assertIn(str(error), str(caught.exception))
        self.assertIn(sys.executable, str(caught.exception))
        self.assertIs(caught.exception.__cause__, error)

    def test_dpd_uses_actual_ofdm_generation_without_mocking_dependency_imports(self):
        from link_sim.algorithm.pa_dpd import PADPDWidget
        from link_sim.algorithm.pa_simulation_core import PAModel
        rate = parameters(50e6, 3)["sampleRate"]
        config = dict(pa=PAModel([[2.]], "memPoly", 1, sample_rate=860.16e6), dpd=None,
            generated=True, bandwidth=50e6, oversampling=3, rate=rate, amplitude=.2,
            count=2048, gain=2., model_type="memPoly", memory=1, degree=1, cross_depth=0,
            iterations=1, measured_pair=None, peak_limit=None, main_bandwidth=50e6,
            adjacent_bandwidth=50e6, adjacent_offset=50e6, nfft=4096, reference_load=100.)
        dpd, result, reference = PADPDWidget.compute(config, True, lambda _: None)
        self.assertEqual(dpd.role, "predistorter")
        self.assertEqual(result["sampleRate"], rate)
        self.assertTrue(np.isfinite(result["y1"]).all())
        self.assertLess(result["metrics"]["after"]["evmPercent"], 1e-8)
        self.assertGreater(reference["numSymbols"], 0)


if __name__ == "__main__":
    unittest.main()
