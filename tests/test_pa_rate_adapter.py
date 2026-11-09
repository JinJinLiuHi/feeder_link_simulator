"""PA/DPD coefficients use input sample delays irrespective of recorded rate."""

import unittest
import numpy as np

from link_sim.algorithm.pa_dpd_core import signature, train_ila, evaluate_dpd
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

    def test_all_rates_use_exact_saved_coefficients_and_sample_delays(self):
        rng = np.random.default_rng(5)
        x = .2*(rng.normal(size=1024)+1j*rng.normal(size=1024))
        for model in self.models():
            fingerprint = signature(model)
            expected = model.predict(x)
            for fs in (100e6, 860.16e6, 2.8e9, 1e12):
                with self.subTest(kind=model.model_type, fs=fs):
                    for compute in (simulate, simulate_at_rate):
                        incoming, y, valid = compute(model, x, fs)
                        np.testing.assert_array_equal(incoming, x)
                        np.testing.assert_array_equal(y, expected)
                        self.assertEqual(valid, model.valid_slice(len(x)))
                    self.assertEqual(signature(model), fingerprint)
                    self.assertEqual(model.sample_rate, 860.16e6)

    def test_one_model_sample_is_one_input_sample_and_dpd_is_direct(self):
        x = np.random.default_rng(5).normal(size=256).astype(complex)
        model = PAModel([[1], [.5j]], "memPoly", 1, sample_rate=100e6)
        _, y, valid = simulate_at_rate(model, x, 300e6)
        self.assertEqual(valid, slice(1, len(x)))
        np.testing.assert_array_equal(y[valid], x[1:]+.5j*x[:-1])
        dpd = PAModel([[2, 0, -.5]], "memPoly", 3, sample_rate=100e6, role="predistorter")
        _, y, valid = simulate_at_rate(dpd, x, 280e6)
        self.assertEqual(valid, slice(0, len(x)))
        np.testing.assert_allclose(y, x*(2-.5*abs(x)**2), atol=1e-14)

    def test_dpd_training_and_evaluation_accept_different_recorded_rates(self):
        rng = np.random.default_rng(24)
        x = .1*(rng.normal(size=512)+1j*rng.normal(size=512))
        pa = PAModel([[2.]], "memPoly", 1, sample_rate=860.16e6)
        dpd = PAModel([[1.]], "memPoly", 1, sample_rate=200e6, role="predistorter",
                      fit_info=dict(paSignature=signature(pa)))
        result = evaluate_dpd(pa, dpd, x, 100e6, 2., main_bandwidth=10e6,
                              adjacent_bandwidth=10e6, adjacent_offset=15e6, nfft=256)
        np.testing.assert_array_equal(result['y1'], pa.predict(dpd.predict(x)))
        self.assertEqual(result['sampleRate'], 100e6)
        fitted = train_ila(pa, x[:256], x[256:], 100e6, 2., memory=1, degree=1, iterations=1)
        self.assertEqual(fitted.model.sample_rate, 100e6)
        self.assertEqual(fitted.model.fit_info['paSignature'], signature(pa))
        self.assertEqual(pa.sample_rate, 860.16e6)

    def test_short_capture_is_limited_only_by_discrete_model_memory(self):
        model = next(self.models())
        _, y, valid = simulate_at_rate(model, np.ones(16, dtype=complex), 2.8e9)
        self.assertEqual(valid, slice(2, 16))
        self.assertEqual(len(y), 16)
        with self.assertRaises(ValueError):
            simulate_at_rate(model, np.ones(4, dtype=complex), 2.8e9)

    def test_input_rate_must_still_be_a_valid_positive_value(self):
        model = next(self.models())
        for fs in (0., -1., float('nan'), float('inf')):
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                simulate_at_rate(model, np.ones(16, dtype=complex), fs)


if __name__ == "__main__":
    unittest.main()
