import itertools
import unittest
from unittest.mock import patch

import numpy as np

from clickremover.engine import ProcessingError
from stemstudio.features import split_features, split_repeating


def known_sources(sr=8000, seconds=3):
    time = np.arange(sr * seconds) / sr
    sustained = .1 * np.sin(2 * np.pi * 330 * time) * (.55 + .45 * np.sin(2 * np.pi * .8 * time))
    intermittent = .08 * np.sin(2 * np.pi * 1720 * time) * (np.sin(2 * np.pi * 1.3 * time) > .3)
    return [source.astype(np.float32)[:, None] for source in (sustained, intermittent)]


def snr(estimate, target):
    return float(10 * np.log10(np.mean(target.astype(np.float64) ** 2)
                              / max(np.mean((estimate - target).astype(np.float64) ** 2), 1e-15)))


class FeatureSeparationTests(unittest.TestCase):
    def test_overlapping_known_sources_are_actually_separated(self):
        sources = known_sources()
        mixture = sum(sources)
        tracks = split_features(mixture, 8000, components=2)
        estimates = [audio for _, audio in tracks]
        best = max(itertools.permutations(estimates),
                   key=lambda order: sum(snr(a, b) for a, b in zip(order, sources)))
        for estimate, target in zip(best, sources):
            self.assertGreater(snr(estimate, target), 12)
            self.assertGreater(snr(estimate, target) - snr(mixture, target), 10)
        np.testing.assert_allclose(sum(estimates), mixture, atol=3e-8, rtol=0)
        # Removing one learned candidate must substantially change the mixture.
        self.assertGreater(float(np.linalg.norm(mixture - best[0])), .5)

    def test_stereo_antiphase_preserves_channels_and_reconstruction(self):
        mono = sum(known_sources(seconds=1))
        mixture = np.concatenate((mono, -mono), axis=1)
        tracks = split_features(mixture, 8000, components=3)
        for _, track in tracks:
            self.assertEqual(track.shape, mixture.shape)
            np.testing.assert_allclose(track[:, 0], -track[:, 1], atol=1e-7)
        np.testing.assert_allclose(sum(track for _, track in tracks), mixture, atol=5e-8, rtol=0)

    def test_short_signal_and_fractional_frame_are_aligned(self):
        rng = np.random.default_rng(53)
        for length in (64, 701):
            mixture = (rng.normal(size=(length, 1)) * .02).astype(np.float32)
            tracks = split_features(mixture, 8000, components=2)
            for _, track in tracks:
                self.assertEqual(track.shape, mixture.shape)
                self.assertTrue(np.isfinite(track).all())
            np.testing.assert_allclose(sum(track for _, track in tracks), mixture, atol=2e-8)

    def test_invalid_input_and_silence_are_rejected(self):
        for audio, rate, count in ((np.zeros((100, 1)), 8000, 2),
                                    (np.ones((10, 1)), 8000, 2),
                                    (np.full((100, 1), np.nan), 8000, 2),
                                    (np.ones((100, 3)), 8000, 2),
                                    (np.ones((100, 1)), 7999, 2),
                                    (np.ones((100, 1)), 8000, True),
                                    (np.ones((100, 1)), 8000, 9)):
            with self.subTest(rate=rate, count=count, shape=audio.shape):
                with self.assertRaises(ProcessingError):
                    split_features(audio, rate, count)

    def test_cancel_is_checked_during_training(self):
        cancel = [False]
        def progress(value, message):
            if value >= 10:
                cancel[0] = True
        with self.assertRaisesRegex(ProcessingError, "取消"):
            split_features(sum(known_sources(seconds=1)), 8000, 2,
                           progress=progress, cancelled=lambda: cancel[0])
        with self.assertRaisesRegex(ProcessingError, "取消"):
            split_repeating(sum(known_sources(seconds=1)), 8000, cancelled=lambda: True)

    def test_memory_budget_precedes_spectrogram_allocation(self):
        large = np.full((8000 * 400, 2), .01, dtype=np.float32)
        with patch("stemstudio.features.signal.stft") as stft:
            with self.assertRaisesRegex(ProcessingError, "内存"):
                split_features(large, 8000, 8)
            stft.assert_not_called()

    def test_repeating_extraction_has_known_ground_truth_and_sums_back(self):
        from test_neural import accented_mixture
        mixture, clean, _, _, rate = accented_mixture()
        tracks = split_repeating(mixture, rate)
        candidate, remainder = [track for _, track in tracks]
        np.testing.assert_allclose(candidate + remainder, mixture, atol=3e-8, rtol=0)
        before = np.mean((mixture - clean) ** 2)
        after = np.mean((remainder - clean) ** 2)
        self.assertLess(after, before * .5)
        self.assertGreater(float(np.linalg.norm(candidate)), 1.)

    def test_unrepeated_tone_is_not_reported_as_successful_metronome(self):
        time = np.arange(8000 * 3) / 8000
        tone = (.05 * np.sin(2 * np.pi * 400 * time)).astype(np.float32)[:, None]
        with self.assertRaises(ProcessingError):
            split_repeating(tone, 8000)


if __name__ == "__main__":
    unittest.main()
