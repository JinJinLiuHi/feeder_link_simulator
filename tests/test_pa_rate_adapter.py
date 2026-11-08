"""Physical delay preservation as PA/DPD evaluation sample rates change."""

import unittest
import numpy as np

from link_sim.algorithm.pa_dpd_core import signature
from link_sim.algorithm.pa_rate_adapter import simulate_at_rate
from link_sim.algorithm.pa_simulation_core import PAModel, simulate


class PARateAdapterTests(unittest.TestCase):
    def models(self):
        rng = np.random.default_rng(51)
        for kind, columns, cross in (("memPoly", 4, 0), ("ctMemPoly", 10, 0), ("gmp", 10, 1)):
            coefficients = .1*(rng.normal(size=(3, columns))+1j*rng.normal(size=(3, columns)))
            yield PAModel(coefficients, kind, 4, cross, sample_rate=860.16e6)

    def test_native_rate_is_identical_to_original_simulator(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=1024)+1j*rng.normal(size=1024)
        for model in self.models():
            with self.subTest(kind=model.model_type):
                _, expected, original_valid = simulate(model, x, model.sample_rate)
                _, actual, valid = simulate_at_rate(model, x, model.sample_rate)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(valid, original_valid)

    def test_fractional_delays_match_analytic_continuous_input(self):
        fs = 2.8e9
        def source(t):
            return .3*np.exp(2j*np.pi*70e6*t)+.1*np.exp(-2j*np.pi*140e6*t)
        x = source(np.arange(8192)/fs)
        for model in self.models():
            with self.subTest(kind=model.model_type):
                fingerprint = signature(model)
                _, y, valid = simulate_at_rate(model, x, fs)
                t = np.arange(valid.start, valid.stop)/fs
                expected = np.zeros(len(t), dtype=complex)
                for row, col, delay, envelope_delay, power in model.terms():
                    expected += (model.coefficients[row, col]*source(t-delay/model.sample_rate)
                                 *abs(source(t-envelope_delay/model.sample_rate))**power)
                np.testing.assert_allclose(y[valid], expected, rtol=4e-5, atol=2e-6)
                self.assertEqual(signature(model), fingerprint)
                self.assertEqual(model.sample_rate, 860.16e6)
                self.assertTrue(np.all(y[:valid.start] == 0))
                self.assertTrue(np.all(y[valid.stop:] == 0))

    def test_integer_grid_and_memoryless_dpd_are_exact(self):
        x = np.random.default_rng(5).normal(size=256).astype(complex)
        model = PAModel([[1], [.5j]], "memPoly", 1, sample_rate=100e6)
        _, y, valid = simulate_at_rate(model, x, 300e6)
        self.assertEqual(valid, slice(3, len(x)))
        np.testing.assert_array_equal(y[valid], x[3:]+.5j*x[:-3])
        dpd = PAModel([[2, 0, -.5]], "memPoly", 3, sample_rate=100e6, role="predistorter")
        _, y, valid = simulate_at_rate(dpd, x, 280e6)
        self.assertEqual(valid, slice(0, len(x)))
        np.testing.assert_allclose(y, x*(2-.5*abs(x)**2), atol=1e-14)

    def test_coarser_grid_preserves_analytic_input_and_physical_delays(self):
        fs = 100e6
        def source(t):
            return .3*np.exp(2j*np.pi*8e6*t)+.1*np.exp(-2j*np.pi*11e6*t)
        x = source(np.arange(4096)/fs)
        for model in self.models():
            with self.subTest(kind=model.model_type):
                _, y, valid = simulate_at_rate(model, x, fs)
                t = np.arange(valid.start, valid.stop)/fs
                expected = np.zeros(len(t), dtype=complex)
                for row, col, delay, envelope_delay, power in model.terms():
                    expected += (model.coefficients[row, col]*source(t-delay/model.sample_rate)
                                 *abs(source(t-envelope_delay/model.sample_rate))**power)
                np.testing.assert_allclose(y[valid], expected, rtol=4e-5, atol=2e-6)

    def test_short_capture_is_rejected(self):
        model = next(self.models())
        with self.assertRaises(ValueError):
            simulate_at_rate(model, np.ones(16, dtype=complex), 2.8e9)


if __name__ == "__main__":
    unittest.main()
