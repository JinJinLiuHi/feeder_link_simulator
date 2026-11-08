"""Complex-tone calibration and sign convention for exported/displayed spectra."""

import unittest
import numpy as np

from link_sim.algorithm.waveform_spectrum import amplitude_spectrum


class WaveformSpectrumTests(unittest.TestCase):
    def test_signed_tones_and_absolute_amplitude(self):
        n, fs = 4096, 64e6
        t = np.arange(n)/fs
        x = .25*np.exp(2j*np.pi*8e6*t) + .1*np.exp(-2j*np.pi*4e6*t)
        f, db = amplitude_spectrum(x, fs)
        self.assertEqual(f[np.argmax(db)], 8e6)
        self.assertAlmostEqual(db[np.argmin(abs(f-8e6))], 20*np.log10(.25), places=10)
        self.assertAlmostEqual(db[np.argmin(abs(f+4e6))], -20., places=10)
        self.assertEqual(len(f), n)
        self.assertEqual(f[n//2], 0.)

    def test_dc_is_preserved_and_zero_signal_is_finite(self):
        f, db = amplitude_spectrum(np.full(1024, .2+.2j), 100e6)
        self.assertEqual(f[np.argmax(db)], 0.)
        self.assertAlmostEqual(db.max(), 20*np.log10(abs(.2+.2j)), places=10)
        _, zero_db = amplitude_spectrum(np.zeros(1024, dtype=complex), 100e6)
        self.assertTrue(np.isfinite(zero_db).all())


if __name__ == "__main__":
    unittest.main()
