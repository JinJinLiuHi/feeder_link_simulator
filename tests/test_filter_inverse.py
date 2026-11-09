"""Paired converter filtering preserves timing and has a bounded inverse."""

import unittest
import numpy as np
from scipy import signal

from link_sim.algorithm.device_models import DeviceModel, FILTER_DEFAULTS
from link_sim.algorithm.filter_inverse import inverse_taps
from link_sim.algorithm.waveform_chain import run_graph
from link_sim.algorithm.framed_link import business_output
from waveform_fixtures import configured_link


def lowpass(h=(.75, .25), fs=2.8e9):
    return DeviceModel("filter", dict(FILTER_DEFAULTS, impl="外部系数", fs=fs),
                       {"h": np.asarray(h)}, "配对低通").validated()


class FilterInverseTests(unittest.TestCase):
    def test_inverse_restores_signal_with_known_delay_and_keeps_zero_finite(self):
        rng = np.random.default_rng(70)
        x = rng.normal(size=3000)+1j*rng.normal(size=3000)
        model = lowpass()
        h, delay = inverse_taps(model)
        y = np.convolve(np.convolve(x, h), model.arrays["h"])
        np.testing.assert_allclose(y[delay:delay+len(x)], x, atol=.003)
        # A true zero at Nyquist cannot be inverted. The response must stay bounded.
        h, _ = inverse_taps(lowpass((.5, .5)))
        self.assertTrue(np.isfinite(h).all())
        self.assertLessEqual(np.max(abs(np.fft.fft(h, 16384))), 10.01)
        # Measured frequency responses are stored by the library as 4096 taps.
        h, delay = inverse_taps(lowpass(np.r_[.75, .25, np.zeros(4094)]))
        self.assertTrue(np.isfinite(h).all())
        self.assertAlmostEqual(sum(h), 1., delta=.003)

    def test_inverse_uses_iir_recurrence_response(self):
        model = DeviceModel("filter", dict(FILTER_DEFAULTS, impl="IIR"),
                            {"b": np.array([.75]), "a": np.array([1., -.25])}).validated()
        h, delay = inverse_taps(model)
        w = np.linspace(0, np.pi, 1000)
        _, a = signal.freqz(h, worN=w)
        _, b = signal.freqz(model.arrays["b"], model.arrays["a"], worN=w)
        np.testing.assert_allclose(a*b*np.exp(1j*w*delay), 1., atol=.002)

    def test_converter_pairs_order_models_and_complete_frame_tails(self):
        stages, edges, _ = configured_link()
        by_key = {s.key: s for s in stages}
        for prefix, converter, suffix in (("tx", "dac", "dac"), ("rx", "rx_adc", "adc")):
            pre, lp = f"A_{prefix}_precomp", f"A_{suffix}_lowpass"
            self.assertIn((pre, f"A_{converter}"), edges)
            self.assertIn((f"A_{converter}", lp), edges)
            self.assertEqual(by_key[pre].settings["paired_node"], lp)
            # Only one side loaded: execution resolves the other without mutating it.
            by_key[lp].model = lowpass()
        by_key["A_channel"].settings["snr_db"] = 200.
        outputs = run_graph(stages, edges)
        self.assertIsNone(by_key["A_tx_precomp"].model)
        self.assertGreater(len(outputs["A_dac_lowpass"].x), len(outputs["A_rrc_tx"].x))
        np.testing.assert_array_equal(outputs["A_sink"].decoded_bits, outputs["A_frame_source"].context.bits)
        pa = business_output(outputs["A_pa"], "pa")
        source = business_output(outputs["A_frame_source"], "frame_source")
        self.assertEqual(len(pa.x), 2048)
        self.assertLess(np.linalg.norm(pa.x-source.x)/np.linalg.norm(source.x), .01)
        by_key["A_tx_precomp"].model = lowpass((.8, .2))
        with self.assertRaisesRegex(ValueError, "同一个低通模型"):
            run_graph(stages, edges)


if __name__ == "__main__":
    unittest.main()
