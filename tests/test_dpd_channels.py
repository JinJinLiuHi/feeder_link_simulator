"""Adjacent-channel integration ranges and actionable configuration failures."""

import unittest
import numpy as np
from link_sim.algorithm.pa_dpd_core import channel_acpr, validate_channel_bands


class DPDChannelTests(unittest.TestCase):
    def test_50mhz_channels_need_150mhz_sampling_and_do_not_overlap(self):
        with self.assertRaisesRegex(ValueError, r"(?s)61\.44.*150"):
            validate_channel_bands(61.44e6, 50e6, 50e6, 50e6)
        validate_channel_bands(184.32e6, 50e6, 50e6, 50e6)
        validate_channel_bands(150e6, 50e6, 50e6, 50e6)
        with self.assertRaisesRegex(ValueError, "主邻道重叠.*至少需要 50"):
            validate_channel_bands(184.32e6, 50e6, 50e6, 30e6)
        with self.assertRaisesRegex(ValueError, "主信道带宽.*超过采样率"):
            validate_channel_bands(20e6, 50e6, 50e6, 50e6)

    def test_known_main_and_adjacent_tone_powers(self):
        fs, count = 192e6, 6144
        time = np.arange(count)/fs
        x = 1.+.1*np.exp(-2j*np.pi*50e6*time)+.2*np.exp(2j*np.pi*50e6*time)
        lower, upper = channel_acpr(x, fs, 50e6, 50e6, 50e6, nfft=count)
        self.assertAlmostEqual(lower, -20., places=9)
        self.assertAlmostEqual(upper, 20*np.log10(.2), places=9)


if __name__ == "__main__":
    unittest.main()
