import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.io import wavfile

from clickremover.engine import ProcessingError, _decode
from stemstudio.project import ProjectStore, validate_controls
from stemstudio.service import StudioService, consistent_stems
from stemstudio.server import StudioHTTPServer


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ProjectStore(Path(self.temp.name))
        self.sr = 16000
        t = np.arange(self.sr) / self.sr
        self.piano = np.column_stack((.1 * np.sin(2 * np.pi * 440 * t),
                                     .08 * np.sin(2 * np.pi * 440 * t))).astype(np.float32)
        self.clicks = np.zeros_like(self.piano)
        self.clicks[::800] = .25
        self.audio = self.piano + self.clicks
        self.data = self.store.create(self.audio, self.sr, "示例.m4a")
        self.sid = self.data["id"]
        self.data = self.store.commit_stems(self.sid, [("琴", self.piano), ("拍", self.clicks)], "truth")

    def test_mix_identity_mute_gain_and_subset(self):
        tracks = self.data["tracks"]
        np.testing.assert_allclose(self.store.mix(self.data, tracks), self.audio, atol=1e-7)
        settings = [dict(tracks[0], gain=.6), dict(tracks[1], muted=True)]
        np.testing.assert_allclose(self.store.mix(self.data, settings), self.piano * .6, atol=1e-7)
        np.testing.assert_allclose(self.store.mix(self.data, [tracks[1]]), self.clicks, atol=1e-7)

    def test_subdivision_inherits_controls_undo_and_reload(self):
        settings = [dict(t, gain=.42, muted=True) for t in self.data["tracks"]]
        self.store.update(self.sid, settings)
        divided = self.store.commit_stems(self.sid, [("A", self.piano * .7), ("B", self.piano * .3)],
                                          "features", parent_id=settings[0]["id"])
        self.assertEqual(len(divided["tracks"]), 3)
        self.assertTrue(all(t["muted"] and t["gain"] == .42 for t in divided["tracks"]))
        restored = self.store.undo(self.sid)
        self.assertEqual(restored["tracks"], settings)
        self.assertEqual(ProjectStore(Path(self.temp.name)).load(self.sid)["tracks"], settings)

    def test_invalid_results_and_controls_do_not_replace_project(self):
        original = self.store.load(self.sid)
        with self.assertRaises(ProcessingError):
            self.store.commit_stems(self.sid, [("x", self.piano), ("y", self.piano)], "broken")
        self.assertEqual(self.store.load(self.sid), original)
        for values in ([], [{"id": "missing"}], [dict(original["tracks"][0], gain=float("nan"))],
                       [dict(original["tracks"][0], gain=-1)], [original["tracks"][0]] * 2):
            with self.assertRaises(ProcessingError):
                validate_controls(original["tracks"], values, subset=True)

    def test_neural_residual_restores_original(self):
        estimates = {"piano": self.piano * .92, "vocals": self.clicks * .8}
        stems = consistent_stems(self.audio, estimates)
        self.assertEqual(len(stems), 3)
        np.testing.assert_allclose(sum(t for _, t in stems), self.audio, atol=1e-7)

    def test_practice_pipeline_feeds_remainder_to_network_without_double_counting(self):
        service = StudioService(self.store.root)
        service.active = self.sid
        with patch("stemstudio.features.split_repeating", return_value=[("click", self.clicks), ("rest", self.piano)]), \
             patch("stemstudio.demucs_backend.separate_demucs", return_value={"piano": self.piano * .9, "vocals": self.piano * .02}) as model:
            service.separate("practice")
            service.worker.join(timeout=10)
        self.assertIsNone(service.error)
        np.testing.assert_array_equal(model.call_args.args[0], self.piano)
        data = self.store.load(self.sid)
        np.testing.assert_allclose(self.store.mix(data, data["tracks"]), self.audio, atol=1e-7)
        settings = [dict(t, muted=(i == 0)) for i, t in enumerate(data["tracks"])]
        np.testing.assert_allclose(self.store.mix(data, settings), self.piano, atol=1e-7)

    def test_corrupt_project_does_not_break_listing_or_recovery(self):
        good_id = self.sid
        bad = self.store.create(self.audio, self.sr, "bad")
        broken = dict(bad)
        del broken["name"]
        self.store.save(broken)
        self.assertEqual([p["id"] for p in self.store.list_projects()], [good_id])
        service = StudioService(self.store.root)
        service.active = bad["id"]
        state = service.state()
        self.assertIsNone(state["session"])
        self.assertTrue(state["error"])
        self.assertEqual(service.open(good_id)["id"], good_id)

    def test_import_export_errors_are_visible_and_preserve_current_project(self):
        service = StudioService(self.store.root)
        service.active = self.sid
        with patch.object(service.store, "import_file", side_effect=OSError("decode failed")):
            with self.assertRaises(OSError):
                service.import_file(Path("none"))
        self.assertEqual(service.state()["error"], "decode failed")
        self.assertFalse(service.busy)
        with patch.object(service.store, "export", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                service.export(self.data["tracks"], "wav", "mix")
        state = service.state()
        self.assertEqual(state["error"], "disk full")
        self.assertFalse(state["busy"])
        self.assertEqual(state["session"]["id"], self.sid)

    def test_formats_roundtrip_and_zip_preserves_float_tracks(self):
        settings = [dict(t, muted=(i == 1)) for i, t in enumerate(self.data["tracks"])]
        for fmt in ("wav", "flac", "mp3"):
            result = self.store.export(self.sid, settings, fmt, "mix")
            path = Path(result["path"])
            self.assertTrue(path.exists())
            decoded, sr = _decode(path)
            self.assertEqual(sr, self.sr)
            self.assertEqual(decoded.shape, self.audio.shape)
            # MP3 is lossy (including encoder amplitude/filter changes); require
            # >20 dB reconstruction SNR. Lossless formats must retain PCM precision.
            mse = float(np.mean((decoded - self.piano) ** 2))
            limit = float(np.mean(self.piano ** 2)) * .01 if fmt == "mp3" else 1e-12
            self.assertLess(mse, limit)
        result = self.store.export(self.sid, settings, "wav", "stems")
        with zipfile.ZipFile(result["path"]) as archive:
            self.assertEqual(len(archive.namelist()), 3)
            for name in archive.namelist():
                if name.endswith(".wav"):
                    with archive.open(name) as stream:
                        rate, samples = wavfile.read(stream)
                    self.assertEqual(samples.dtype, np.float32)
                    self.assertEqual(rate, self.sr)

    def test_export_headroom_is_common_gain_without_overwriting(self):
        settings = [dict(t, gain=4) for t in self.data["tracks"]]
        one = self.store.export(self.sid, settings, "wav", "mix")
        two = self.store.export(self.sid, settings, "wav", "mix")
        self.assertNotEqual(one["path"], two["path"])
        self.assertLess(one["scale"], 1)
        decoded, _ = _decode(Path(one["path"]))
        np.testing.assert_allclose(decoded, self.audio * 4 * one["scale"], atol=2e-7)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = StudioService(Path(self.temp.name))
        self.server = StudioHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.headers), response.read()
        finally:
            conn.close()

    def test_local_import_state_audio_range_and_request_bounds(self):
        path = Path(self.temp.name) / "input.wav"
        wavfile.write(path, 16000, np.zeros((1600, 1), dtype=np.float32))
        status, _, body = self.request("POST", "/api/import", path.read_bytes(), {"X-Filename": "input.wav"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        status, headers, body = self.request("GET", data["original_url"], headers={"Range": "bytes=0-43"})
        self.assertEqual(status, 206)
        self.assertEqual(len(body), 44)
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-43/"))
        self.assertEqual(self.request("GET", data["original_url"], headers={"Range": "bytes=99999999-"})[0], 416)
        self.assertEqual(self.request("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/open", b"{}", {"Content-Type": "application/json", "Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.request("GET", "/audio/../../AGENTS.md")[0], 404)
        self.assertEqual(self.request("POST", "/api/session", b"{}", {"Content-Type": "application/json"})[0], 400)

    def test_failed_job_retains_project_and_reports_error(self):
        audio = np.ones((1600, 1), dtype=np.float32) * .01
        data = self.service.store.create(audio, 16000, "test")
        self.service.remember(data["id"])
        with patch("stemstudio.features.split_features", side_effect=RuntimeError("test failure")):
            self.service.separate("features", 2)
            self.service.worker.join(timeout=10)
        state = self.service.state()
        self.assertFalse(state["busy"])
        self.assertEqual(state["error"], "test failure")
        self.assertEqual(state["session"]["tracks"], [])


if __name__ == "__main__":
    unittest.main()
