import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from clickremover.engine import ProcessConfig, ProcessingError, _decode, process_file, remove_samples


def make_mixture(stereo=True, with_click=True):
    """Known piano-like tones plus a repeated band-limited click with timing jitter."""
    sr = 16000
    duration = 9.0
    rng = np.random.default_rng(129)
    n = int(sr * duration)
    clean = np.zeros(n, dtype=np.float32)
    # Some notes begin close to the click, some between beats.
    notes = [.45, 1.12, 1.55, 2.09, 2.8, 3.18, 3.74, 4.30, 4.93, 5.40, 6.1, 6.68, 7.2, 7.85]
    for i, start in enumerate(notes):
        length = int(.75 * sr)
        t = np.arange(length) / sr
        frequency = [220, 277.18, 329.63, 392, 493.88, 587.33][i % 6]
        envelope = (1 - np.exp(-t / .004)) * np.exp(-t / .32)
        tone = sum(np.sin(2 * np.pi * frequency * harmonic * t + .13 * harmonic) / harmonic ** 1.2
                   for harmonic in range(1, 12) if harmonic * frequency < sr / 2)
        attack = rng.normal(0, 1, length) * np.exp(-t / .009) * .05
        piece = (.07 * tone * envelope + attack).astype(np.float32)
        offset = int(start * sr)
        end = min(offset + length, n)
        clean[offset:end] += piece[:end-offset]
    click_t = np.arange(int(.09 * sr)) / sr
    click = rng.normal(size=len(click_t)) * np.exp(-click_t / .017)
    click = signal.sosfilt(signal.butter(3, 1700, "highpass", fs=sr, output="sos"), click)
    click = (.12 * click / np.max(np.abs(click))).astype(np.float32)
    interference = np.zeros(n, dtype=np.float32)
    times = .5 + np.arange(15) * .55 + rng.uniform(-.008, .008, 15)
    for i, time in enumerate(times):
        start = int(time * sr)
        interference[start:start+len(click)] += click * (0.8 + .4 * (i % 3) / 2)
    if stereo:
        clean = np.column_stack((clean, .83 * clean))
        interference = np.column_stack((interference, .72 * interference))
    else:
        clean, interference = clean[:, None], interference[:, None]
    return clean + (interference if with_click else 0), clean, interference, times, click, sr


class RemovalTests(unittest.TestCase):
    def test_stereo_mixture_reduces_true_error_and_preserves_other_times(self):
        mixture, clean, click, times, _, sr = make_mixture(stereo=True)
        output, report = remove_samples(mixture, sr, ProcessConfig(strength=.8))
        self.assertEqual(output.shape, mixture.shape)
        self.assertGreaterEqual(report["processed_count"], 12)
        before = float(np.mean((mixture - clean) ** 2))
        after = float(np.mean((output - clean) ** 2))
        self.assertLess(after, before * .65, (before, after))
        self.assertTrue(np.isfinite(output).all())
        # No event window touches this gap; processing must not globally reshape it.
        np.testing.assert_array_equal(output[int(8.65 * sr):], mixture[int(8.65 * sr):])

    def test_mono_reference_assisted_detection(self):
        mixture, clean, _, _, click, sr = make_mixture(stereo=False)
        ref = np.zeros((int(.3 * sr), 1), dtype=np.float32)
        ref[int(.1 * sr):int(.1 * sr) + len(click), 0] = click
        output, report = remove_samples(mixture, sr, ProcessConfig(strength=.8), reference=(ref, sr))
        self.assertGreaterEqual(report["processed_count"], 12)
        self.assertLess(np.mean((output - clean) ** 2), .7 * np.mean((mixture - clean) ** 2))

    def test_clicks_overlapping_piano_attacks(self):
        mixture, clean, _, times, _, sr = make_mixture()
        for i, start in enumerate(times):
            t = np.arange(int(.4 * sr)) / sr
            fundamental = [261.63, 329.63, 392, 440, 523.25][i % 5]
            env = (1 - np.exp(-t / .003)) * np.exp(-t / .22)
            note = .12 * sum(np.sin(2 * np.pi * fundamental * k * t + i * .37 * k) / k ** 1.2
                             for k in range(1, 12) if k * fundamental < sr / 2) * env
            pos = int(start * sr)
            end = min(pos + len(note), len(mixture))
            for array in (mixture, clean):
                array[pos:end, 0] += note[:end-pos]
                array[pos:end, 1] += .83 * note[:end-pos]
        output, report = remove_samples(mixture, sr, ProcessConfig(strength=.8))
        self.assertGreaterEqual(report["processed_count"], 10)
        self.assertLess(np.mean((output - clean) ** 2), .5 * np.mean((mixture - clean) ** 2))

    def test_clean_nonrepeating_piano_is_not_modified(self):
        mixture, _, _, _, _, sr = make_mixture(with_click=False)
        with self.assertRaises(ProcessingError):
            remove_samples(mixture, sr, ProcessConfig())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "输入.wav"
            output = Path(directory) / "输出.wav"
            source.write_bytes(b"source bytes")
            output.write_bytes(b"existing bytes")
            with self.assertRaises(ProcessingError):
                process_file(str(source), str(output))
            self.assertEqual(source.read_bytes(), b"source bytes")
            self.assertEqual(output.read_bytes(), b"existing bytes")

    def test_real_file_round_trip_and_csv(self):
        mixture, _, _, _, click, sr = make_mixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "钢琴录音.wav"
            output = Path(directory) / "去节拍.wav"
            wavfile.write(source, sr, (np.clip(mixture, -1, 1) * 32767).astype(np.int16))
            original_bytes = source.read_bytes()
            report = process_file(str(source), str(output), config=ProcessConfig())
            decoded, output_sr = _decode(output)
            self.assertEqual(output_sr, sr)
            self.assertEqual(decoded.shape, mixture.shape)
            self.assertGreaterEqual(report["processed_count"], 12)
            self.assertTrue(Path(report["event_csv"]).is_file())
            self.assertEqual(source.read_bytes(), original_bytes)
            reference = Path(directory) / "参考节拍.wav"
            reference_audio = np.zeros(int(.3 * sr), dtype=np.float32)
            reference_audio[int(.1 * sr):int(.1 * sr) + len(click)] = click
            wavfile.write(reference, sr, (reference_audio * 32767).astype(np.int16))
            output2 = Path(directory) / "参考处理.wav"
            report2 = process_file(str(source), str(output2),
                                   config=ProcessConfig(reference_path=str(reference), export_timestamps=False))
            self.assertGreaterEqual(report2["processed_count"], 12)
            self.assertIsNone(report2["event_csv"])
            self.assertTrue(output2.is_file())

    def test_uncertain_clean_file_creates_no_output(self):
        clean, _, _, _, _, sr = make_mixture(with_click=False)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "钢琴.wav"
            output = Path(directory) / "输出.wav"
            wavfile.write(source, sr, (np.clip(clean, -1, 1) * 32767).astype(np.int16))
            with self.assertRaises(ProcessingError):
                process_file(str(source), str(output))
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.stem + "_节拍事件.csv").exists())

    def test_invalid_parameters_are_rejected(self):
        mixture, _, _, _, _, sr = make_mixture()
        with self.assertRaises(ProcessingError):
            remove_samples(mixture, sr, ProcessConfig(strength=float("nan")))
        with self.assertRaises(ProcessingError):
            remove_samples(mixture, sr, ProcessConfig(min_bpm=200, max_bpm=50))


if __name__ == "__main__":
    unittest.main()
