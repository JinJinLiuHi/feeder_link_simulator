"""Shared frame definition, explicit business area, hex header and seed contract."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from link_sim.algorithm.frame_config import frame_plan, frame_summary, header_bits
from link_sim.algorithm.framed_link import frame_source, business_output
from link_sim.algorithm.waveform_chain import default_link, save_chain, load_chain


class FrameSourceConfigTests(unittest.TestCase):
    def test_default_header_is_msb_first_32_bpsk_symbols_and_padding_is_complex_zero(self):
        wave = frame_source(dict(seed=7))
        info, framer = wave.metrics, wave.context.framer
        self.assertEqual(info["total_symbols"], 1024+4*(32+1024))
        self.assertEqual(info["information_bits"], 4*512)
        self.assertEqual(info["coded_bits"], 4*1024)
        self.assertEqual(info["zero_padding_symbols"], 4*512)
        self.assertEqual(len(framer.header_symbols), 32)
        bits = np.array([int(x) for x in f"{int('1ACFFC1D', 16):032b}"])
        from link_sim.algorithm.burst_link.modulation import create
        expected = create("BPSK").modulate(torch.tensor(bits, dtype=torch.uint8)).numpy()
        np.testing.assert_array_equal(framer.header_symbols.numpy(), expected)
        for group, (start, stop) in zip(framer.codeword_spans, framer.payload_spans):
            a, b = group[0]
            self.assertEqual(a, start)
            self.assertEqual(stop-start, 1024)
            np.testing.assert_array_equal(wave.x[b:stop], np.zeros(512, complex))
        payload = business_output(wave, "frame_source")
        self.assertEqual(len(payload.x), 2048)
        np.testing.assert_allclose(abs(payload.x), 1., atol=1e-6)
        np.testing.assert_allclose(abs(payload.x.real), 1/np.sqrt(2), atol=1e-6)
        self.assertAlmostEqual(np.mean(abs(payload.x)**2), 1., places=6)
        self.assertEqual(payload.metrics["omitted_padding_symbols"], 2048)
        np.testing.assert_array_equal(framer.decode(torch.as_tensor(wave.x)).numpy(), wave.context.bits)
        data = wave.context.bits.reshape(4, 512)
        self.assertFalse(np.array_equal(data[0], data[1]))

    def test_observation_excludes_padding_by_frame_position_even_if_it_contains_noise(self):
        wave = frame_source(dict(seed=7))
        framer = wave.context.framer
        spans = [span for group in framer.codeword_spans for span in group]
        for (_, stop), (_, end) in zip(spans, framer.payload_spans):
            wave.x[stop:end] = 3+4j
        wave.x[spans[0][0]] = 0.  # A legitimate business value must remain.
        expected = np.concatenate([wave.x[a:b] for a, b in spans])
        observed = business_output(wave, "frame_source")
        np.testing.assert_array_equal(observed.x, expected)
        self.assertEqual(len(observed.x), 2048)
        self.assertEqual(observed.x[0], 0.)

    def test_payload_capacity_is_measured_after_modulation_and_equality_is_allowed(self):
        for size in (512, 513, 1024):
            wave = frame_source(dict(segments=1, payload_symbols=size, seed=17))
            self.assertEqual(wave.metrics["zero_padding_symbols"], size-512)
            self.assertEqual(len(wave.x), 1024+32+size)
        with self.assertRaisesRegex(ValueError, "512.*超过.*511"):
            frame_source(dict(payload_symbols=511))
        for value in (0, -1, 1024.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                frame_summary(dict(payload_symbols=value))

    def test_zero_seed_selects_fresh_seed_and_positive_seed_repeats(self):
        with patch("link_sim.algorithm.framed_link.secrets.randbelow", side_effect=[100, 200]) as choose:
            first, second = frame_source({}), frame_source(dict(seed=0))
            self.assertEqual(choose.call_count, 2)
        self.assertEqual(first.metrics["actual_seed"], 101)
        self.assertEqual(second.context.seed, 201)
        self.assertFalse(np.array_equal(first.context.bits, second.context.bits))
        with patch("link_sim.algorithm.framed_link.secrets.randbelow", side_effect=AssertionError):
            repeat = frame_source(dict(seed=101))
        np.testing.assert_array_equal(repeat.x, first.x)

    def test_hex_header_length_leading_zeros_and_invalid_input(self):
        self.assertEqual(header_bits(dict(header_hex="0x 0001")), [0]*15+[1])
        for value in ("", "XYZ", "12-34", "F"*257, 1234):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "帧头"):
                frame_summary(dict(header_hex=value))
        wave = frame_source(dict(header_hex="0001", seed=3))
        self.assertEqual(wave.context.framer.header_symbol_count, 16)
        self.assertEqual(wave.metrics["known_header_bits"], 1024+4*16)

    def test_shared_configuration_is_portable_and_legacy_plan_still_runs(self):
        stages, edges, positions = default_link()
        source = stages[0]
        source.settings.update(seed=8, header_hex="0001", segments=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"shared.json"
            save_chain(path, stages, edges, positions)
            loaded, _, _ = load_chain(path)
        saved = loaded[0].settings
        self.assertNotIn("words", saved)
        self.assertNotIn("business_frames", saved)
        self.assertEqual(saved["header_hex"], "0001")
        plan = frame_plan(saved)
        self.assertEqual(len(plan), 3)
        self.assertTrue(all(len(row) == 1 and row[0] == plan[0][0] for row in plan))
        np.testing.assert_array_equal(frame_source(saved).x, frame_source(source.settings).x)
        old = dict(segments=1, words=2, seed=3)
        self.assertEqual(len(frame_source(old).x), 1024+64+2*512)
        self.assertEqual(frame_summary(old)["zero_padding_symbols"], 0)

    def test_fastica_training_mask_ignores_padding_without_shortening_outputs(self):
        from link_sim.algorithm.burst_link.xpic_fastica import ComplexFastICA
        rng = np.random.default_rng(42)
        points = np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)
        sources = points[rng.integers(0, 4, (2048, 2))]
        received = sources @ np.array([[1., .2j], [.3, 1.]])
        full = np.r_[received, np.zeros_like(received)]
        mask = torch.arange(4096) < 2048
        a, b = torch.as_tensor(full[:, 0]), torch.as_tensor(full[:, 1])
        ica = ComplexFastICA()
        masked = ica(a, b, start=0, end=4096, training_mask=mask)
        reference = ica(a[:2048], b[:2048], start=0, end=2048)
        self.assertTrue(masked.converged)
        self.assertEqual(len(masked.components[0]), 4096)
        torch.testing.assert_close(masked.unmixing, reference.unmixing)
        torch.testing.assert_close(masked.mean, reference.mean)
        with self.assertRaisesRegex(ValueError, "有效业务样本不足"):
            ica(a, b, start=0, end=4096, training_mask=torch.arange(4096) < 100)


if __name__ == "__main__":
    unittest.main()
