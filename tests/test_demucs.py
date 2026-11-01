from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from stemstudio import demucs_backend as backend


class DemucsValidationTests(unittest.TestCase):
    def test_model_file_must_exist_and_match_hash_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.th"
            with patch.object(backend, "CHECKPOINT", model):
                with self.assertRaisesRegex(backend.SeparationError, "缺少"):
                    backend._validate_checkpoint()
                model.write_bytes(b"invalid checkpoint")
                with self.assertRaisesRegex(backend.SeparationError, "校验"):
                    backend._validate_checkpoint()

    def test_invalid_inputs_and_prestart_cancellation(self):
        with self.assertRaises(ValueError):
            backend.separate_demucs(np.zeros((0, 2)), 8000)
        with self.assertRaises(ValueError):
            backend.separate_demucs(np.zeros((10, 3)), 8000)
        with self.assertRaises(ValueError):
            backend.separate_demucs(np.full((100, 1), np.inf), 8000)
        with self.assertRaises(ValueError):
            backend.separate_demucs(np.zeros((100, 1)), 2)
        with self.assertRaises(InterruptedError):
            backend.separate_demucs(np.zeros((100, 1)), 8000, cancelled=lambda: True)

    @unittest.skipUnless(backend.check_model()["available"], "Demucs checkpoint/dependencies not installed")
    def test_real_model_mono_and_antiphase_stereo_are_finite_and_aligned(self):
        rate = 8000
        t = np.arange(rate // 2) / rate
        mono = (.08 * np.sin(2 * np.pi * 440 * t) * np.exp(-t)).astype(np.float32)[:, None]
        for source in (mono, np.concatenate((mono, -mono), axis=1)):
            with self.subTest(channels=source.shape[1]):
                output = backend.separate_demucs(source, rate)
                self.assertEqual(tuple(output), backend.SOURCES)
                energy = 0.
                for name, track in output.items():
                    self.assertEqual(track.shape, source.shape, name)
                    self.assertEqual(track.dtype, np.float32)
                    self.assertTrue(np.isfinite(track).all(), name)
                    self.assertLess(float(np.max(np.abs(track))), 2., name)
                    energy += float(np.sum(track ** 2))
                self.assertGreater(energy, .01 * float(np.sum(source ** 2)))

    @unittest.skipUnless(backend.check_model()["available"], "Demucs checkpoint/dependencies not installed")
    def test_running_worker_can_be_cancelled_without_resource_warning(self):
        cancelled = [False]
        def progress(value, message):
            if value >= 5:
                cancelled[0] = True
        source = np.random.default_rng(12).normal(0, .01, (8000 * 2, 1)).astype(np.float32)
        start = time.monotonic()
        with self.assertRaises(InterruptedError):
            backend.separate_demucs(source, 8000, progress=progress, cancelled=lambda: cancelled[0])
        self.assertLess(time.monotonic() - start, 30)


if __name__ == "__main__":
    unittest.main()
