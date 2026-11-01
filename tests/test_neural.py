import unittest
from unittest.mock import patch
import importlib.util
import os
import subprocess
import sys

import numpy as np

from clickremover.engine import ProcessConfig, ProcessingError, remove_samples
from test_engine import make_mixture


def accented_mixture():
    mixture, clean, interference, times, _, sr = make_mixture(stereo=True)
    # Independent ground-truth ringing accent on every third click, with timing jitter.
    t = np.arange(int(.16 * sr)) / sr
    accent = (.09 * np.sin(2 * np.pi * 1960 * t) * np.exp(-t / .045)).astype(np.float32)
    for start in times[2::3]:
        pos = int(start * sr)
        end = min(pos + len(accent), len(mixture))
        for channel, amplitude in ((0, 1.), (1, .72)):
            add = amplitude * accent[:end-pos]
            mixture[pos:end, channel] += add
            interference[pos:end, channel] += add
    return mixture, clean, interference, times, sr


class NeuralModeTests(unittest.TestCase):
    def test_regular_neighbors_correct_a_displaced_match_only_with_strong_evidence(self):
        from clickremover.multiclass import _correct_timing
        sr = 16000
        rng = np.random.default_rng(19)
        query = rng.standard_normal(round(.014 * sr)).astype(np.float32)
        pre = round(.006 * sr)
        y = np.zeros(sr * 3, dtype=np.float32)
        correct = round(1.5 * sr)
        y[correct-pre:correct-pre+len(query)] = query
        centers = np.rint(np.array([.5, 1., 1.54, 2., 2.5]) * sr).astype(int)
        scores = np.ones(len(centers)) * .95
        fixed, _, changes = _correct_timing(y, sr, query, pre, centers, scores)
        self.assertEqual(changes, 1)
        self.assertEqual(fixed[2], correct)
        missing, _, changes = _correct_timing(np.zeros_like(y), sr, query, pre, centers, scores)
        self.assertEqual(changes, 0)
        np.testing.assert_array_equal(missing, centers)

    def test_neural_guide_handles_distinct_ringing_accent_and_preserves_gap(self):
        mixture, clean, _, times, sr = accented_mixture()
        # An imperfect piano estimate exercises the neural-guided path deterministically.
        with patch("clickremover.neural.estimate_piano_isolated", return_value=clean * .6) as estimate:
            output, report = remove_samples(mixture, sr, ProcessConfig(method="neural", strength=.95))
        estimate.assert_called_once()
        self.assertEqual(output.shape, mixture.shape)
        self.assertGreaterEqual(report["processed_count"], 12)
        self.assertTrue(np.isfinite(output).all())
        error_before = np.mean((mixture - clean) ** 2)
        error_after = np.mean((output - clean) ** 2)
        self.assertLess(error_after, error_before * .45, (error_before, error_after))
        # No predicted event affects the last gap: avoid reshaping the entire piece.
        np.testing.assert_array_equal(output[int(8.65 * sr):], mixture[int(8.65 * sr):])

    def test_neural_guide_changes_the_result_and_zero_strength_keeps_source(self):
        mixture, clean, _, _, sr = accented_mixture()
        with patch("clickremover.neural.estimate_piano_isolated", return_value=clean * .6):
            guided, _ = remove_samples(mixture, sr, ProcessConfig(method="neural", strength=.95))
            unchanged, _ = remove_samples(mixture, sr, ProcessConfig(method="neural", strength=0))
        with patch("clickremover.neural.estimate_piano_isolated", return_value=np.zeros_like(clean)):
            unguided, report = remove_samples(mixture, sr, ProcessConfig(method="neural", strength=.95))
        self.assertGreater(float(np.max(np.abs(guided - unguided))), 1e-7)
        self.assertTrue(any("保护可能作用很小" in warning for warning in report["warnings"]))
        np.testing.assert_array_equal(unchanged, mixture)

    def test_neural_guidance_supports_mono(self):
        mixture, clean, _, _, sr = accented_mixture()
        mixture = mixture.mean(axis=1, keepdims=True)
        clean = clean.mean(axis=1, keepdims=True)
        with patch("clickremover.neural.estimate_piano_isolated", return_value=clean * .6):
            result, report = remove_samples(mixture, sr, ProcessConfig(method="neural", strength=.95))
        self.assertEqual(result.shape, mixture.shape)
        self.assertGreaterEqual(report["processed_count"], 12)
        self.assertLess(np.mean((result - clean) ** 2), .45 * np.mean((mixture - clean) ** 2))

    def test_neural_failure_does_not_silently_use_classic_method(self):
        mixture, _, _, _, sr = accented_mixture()
        with patch("clickremover.neural.estimate_piano_isolated", side_effect=ProcessingError("test missing model")):
            with self.assertRaisesRegex(ProcessingError, "test missing model"):
                remove_samples(mixture, sr, ProcessConfig(method="neural"))

    def test_invalid_method_is_rejected(self):
        mixture, _, _, _, sr = accented_mixture()
        with self.assertRaises(ProcessingError):
            remove_samples(mixture, sr, ProcessConfig(method="invalid"))

    def test_overlap_add_preserves_identity_and_channel_alignment(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is not installed")
        # Isolate PyTorch from the parent's scientific-computing OpenMP runtime.
        script = '''
from unittest.mock import patch
import numpy as np
import torch
from clickremover.neural import estimate_piano
class Identity(torch.nn.Module):
    def forward(self, tensor):
        return tensor[:, None, :, :]
sr = 44100
original = (np.random.default_rng(15).standard_normal((int(sr * 4.1), 2)) * .01).astype(np.float32)
with patch("clickremover.neural._load_model", return_value=(torch, Identity())):
    restored = estimate_piano(original, sr)
np.testing.assert_allclose(restored, original, atol=1e-7, rtol=1e-5)
'''
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_real_network_loads_and_runs_on_cpu_when_available(self):
        from clickremover.neural import MODEL_DIR, estimate_piano_isolated
        if not (MODEL_DIR / "model.safetensors").is_file() or importlib.util.find_spec("torch") is None:
            self.skipTest("The optional neural model or PyTorch is not installed")
        sr = 44100
        t = np.arange(sr) / sr
        piano_like = (np.sin(2 * np.pi * 440 * t) * .05 * np.exp(-t)).astype(np.float32)
        audio = np.column_stack((piano_like, piano_like * .8))
        estimate = estimate_piano_isolated(audio, sr)
        self.assertEqual(estimate.shape, audio.shape)
        self.assertTrue(np.isfinite(estimate).all())
        self.assertGreater(float(np.max(np.abs(estimate))), 1e-8)
        mono = audio[:, :1]
        mono_estimate = estimate_piano_isolated(mono, sr)
        self.assertEqual(mono_estimate.shape, mono.shape)
        self.assertTrue(np.isfinite(mono_estimate).all())

    def test_real_network_and_separation_on_known_mixture_when_available(self):
        from clickremover.neural import MODEL_DIR
        if not (MODEL_DIR / "model.safetensors").is_file() or importlib.util.find_spec("torch") is None:
            self.skipTest("The optional neural model or PyTorch is not installed")
        mixture, clean, _, _, sr = accented_mixture()
        result, report = remove_samples(mixture, sr, ProcessConfig(method="neural", strength=.95))
        self.assertGreaterEqual(report["processed_count"], 12)
        self.assertLess(np.mean((result - clean) ** 2), .45 * np.mean((mixture - clean) ** 2))


if __name__ == "__main__":
    unittest.main()
