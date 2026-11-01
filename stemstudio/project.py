"""Persist aligned, unnormalised stems and reversible mixer decisions."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import uuid
import zipfile

import numpy as np
from scipy.io import wavfile

from clickremover.engine import ProcessingError, _decode, _program

MAX_DECODED_BYTES = 180 * 1024 * 1024


def safe_name(name: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .")[:90]
    return value or "未命名"


def check_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if (audio.ndim != 2 or audio.shape[1] not in (1, 2) or len(audio) < 1
            or not 8000 <= sr <= 192000 or not np.isfinite(audio).all()):
        raise ProcessingError("无效音频：需要有限值、单声道或立体声、8–192 kHz。")
    return audio


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    """IEEE float WAV retains headroom; never normalise a single stem."""
    if path.exists():
        raise ProcessingError("拒绝覆盖已有音轨文件。")
    with path.open("xb") as stream:
        wavfile.write(stream, sr, np.asarray(audio, dtype=np.float32))
    mono_peak = np.max(np.abs(audio), axis=1)
    edges = np.linspace(0, len(audio), min(len(audio), 2048) + 1).astype(int)
    peaks = [float(np.max(mono_peak[a:b])) for a, b in zip(edges[:-1], edges[1:])]
    atomic_json(path.with_suffix(".peaks.json"), {
        "peaks": peaks, "duration": len(audio) / sr,
    })


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    sr, audio = wavfile.read(path)
    if audio.dtype != np.float32:
        raise ProcessingError("工程音轨不是浮点 WAV，请重新导入原音。")
    if audio.ndim == 1:
        audio = audio[:, None]
    return check_audio(audio, sr), sr


def validate_controls(tracks: list[dict], values: list[dict], *, subset: bool) -> list[dict]:
    if not isinstance(values, list) or not values:
        raise ProcessingError("至少选择一条音轨。")
    known = {t["id"]: t for t in tracks}
    result, seen = [], set()
    for value in values:
        if not isinstance(value, dict):
            raise ProcessingError("音轨参数无效。")
        identifier = value.get("id")
        if identifier not in known or identifier in seen:
            raise ProcessingError("音轨不存在或重复。")
        seen.add(identifier)
        try:
            gain = float(value.get("gain", known[identifier]["gain"]))
        except (ValueError, TypeError):
            raise ProcessingError("音量必须是有限数字。") from None
        if not math.isfinite(gain) or not 0 <= gain <= 4:
            raise ProcessingError("音量范围为 0–4 倍。")
        muted = value.get("muted", known[identifier]["muted"])
        if not isinstance(muted, bool):
            raise ProcessingError("静音状态必须为布尔值。")
        name = str(value.get("name", known[identifier]["name"])).strip()[:100]
        result.append({**known[identifier], "gain": gain, "muted": muted,
                       "name": name or known[identifier]["name"]})
    if not subset and seen != set(known):
        raise ProcessingError("保存混音必须包含所有当前音轨。")
    return result


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def directory(self, identifier: str) -> Path:
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ProcessingError("无效工程编号。")
        return self.root / identifier

    def load(self, identifier: str) -> dict:
        path = self.directory(identifier) / "project.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ProcessingError("工程不存在或工程文件损坏。") from None
        try:
            if data["id"] != identifier or data["version"] != 1:
                raise ValueError()
            if not all(isinstance(data[key], str) for key in ("name", "created")):
                raise ValueError()
            for key in ("samples", "sample_rate", "channels", "revision"):
                if type(data[key]) is not int:
                    raise ValueError()
            if not (data["samples"] > 0 and 8000 <= data["sample_rate"] <= 192000
                    and data["channels"] in (1, 2) and data["revision"] >= 0
                    and math.isfinite(data["duration"])
                    and abs(data["duration"] - data["samples"] / data["sample_rate"]) < .001):
                raise ValueError()
            if not isinstance(data["history"], list) or not isinstance(data["notes"], list):
                raise ValueError()
            for tracks in [data["tracks"]] + [h["tracks"] for h in data["history"]]:
                if not isinstance(tracks, list) or len(tracks) > 32:
                    raise ValueError()
                seen = set()
                for t in tracks:
                    tid = t["id"]
                    if (not isinstance(tid, str) or not re.fullmatch(r"[0-9a-f]{32}", tid)
                            or tid in seen or t["file"] != tid + ".wav"
                            or not isinstance(t["name"], str) or not isinstance(t["method"], str)
                            or not isinstance(t["muted"], bool)
                            or not math.isfinite(t["gain"]) or not 0 <= t["gain"] <= 4):
                        raise ValueError()
                    seen.add(tid)
            for snapshot in data["history"]:
                if not isinstance(snapshot["notes"], list):
                    raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ProcessingError("工程结构损坏，请重新导入原音或打开其他工程。") from None
        return data

    def save(self, data: dict) -> None:
        atomic_json(self.directory(data["id"]) / "project.json", data)

    def list_projects(self) -> list[dict]:
        found = []
        for path in self.root.glob("*/project.json"):
            try:
                item = self.load(path.parent.name)
                found.append({"id": item["id"], "name": item["name"],
                              "created": item["created"], "tracks": len(item["tracks"])})
            except ProcessingError:
                continue
        return sorted(found, key=lambda x: x["created"], reverse=True)

    def public(self, data: dict) -> dict:
        base = f'/audio/{data["id"]}/'
        return {key: data[key] for key in (
            "id", "name", "created", "duration", "sample_rate", "channels", "revision",
        )} | {
            "original_url": base + "original.wav", "can_undo": bool(data.get("history")),
            "tracks": [{**t, "url": base + t["file"],
                        "peak_url": base + Path(t["file"]).with_suffix(".peaks.json").name}
                       for t in data["tracks"]],
            "notes": data.get("notes", []),
        }

    def import_file(self, source: Path, display_name: str | None = None) -> dict:
        audio, sr = _decode(Path(source), max_duration=600)
        if audio.nbytes > MAX_DECODED_BYTES:
            raise ProcessingError("录音解码后超过 180 MiB；请先分段，以便浏览器同步试听。")
        return self.create(audio, sr, display_name or source.name)

    def create(self, audio: np.ndarray, sr: int, name: str) -> dict:
        audio = check_audio(audio, sr)
        identifier = uuid.uuid4().hex
        directory = self.directory(identifier)
        directory.mkdir()
        write_audio(directory / "original.wav", audio, sr)
        data = {"version": 1, "id": identifier, "name": safe_name(Path(name).stem),
                "created": datetime.now(timezone.utc).isoformat(), "sample_rate": sr,
                "channels": audio.shape[1], "samples": len(audio), "duration": len(audio) / sr,
                "revision": 0, "tracks": [], "history": [], "notes": []}
        self.save(data)
        return data

    def original(self, data: dict) -> np.ndarray:
        return read_audio(self.directory(data["id"]) / "original.wav")[0]

    def track_audio(self, data: dict, track: dict) -> np.ndarray:
        filename = track["file"]
        if not re.fullmatch(r"[0-9a-f]{32}\.wav", filename):
            raise ProcessingError("工程音轨文件名无效。")
        audio, sr = read_audio(self.directory(data["id"]) / filename)
        if sr != data["sample_rate"] or audio.shape != (data["samples"], data["channels"]):
            raise ProcessingError("音轨长度、采样率或声道与工程不一致。")
        return audio

    def commit_stems(self, identifier: str, stems: list[tuple[str, np.ndarray]], method: str,
                     *, parent_id: str | None = None, notes: list[str] | None = None) -> dict:
        """Publish a whole successful separation at once; keep old tracks for undo."""
        with self.lock:
            data = self.load(identifier)
            parent = None
            if parent_id:
                parent = next((t for t in data["tracks"] if t["id"] == parent_id), None)
                if parent is None:
                    raise ProcessingError("待拆分音轨已不存在。")
            if not 2 <= len(stems) <= 16:
                raise ProcessingError("分离结果需要 2–16 条音轨。")
            # Validation happens before any files are published.
            for _, audio in stems:
                check_audio(audio, data["sample_rate"])
                if audio.shape != (data["samples"], data["channels"]):
                    raise ProcessingError("分离器返回的音轨与输入不对齐。")
            target = self.track_audio(data, parent) if parent else self.original(data)
            total = sum((a.astype(np.float64) for _, a in stems), start=np.zeros_like(target, dtype=np.float64))
            error = float(np.max(np.abs(total - target)))
            if error > 2e-5:
                raise ProcessingError(f"音轨重建检查未通过（最大采样差 {error:.4g}）。")
            if len(data["tracks"]) + len(stems) - (1 if parent else len(data["tracks"])) > 32:
                raise ProcessingError("单个工程最多 32 条轨，请在新工程中继续拆分。")
            new_tracks = []
            for name, audio in stems:
                track_id = uuid.uuid4().hex
                filename = track_id + ".wav"
                write_audio(self.directory(identifier) / filename, audio, data["sample_rate"])
                new_tracks.append({"id": track_id, "file": filename, "name": name,
                                   "method": method, "gain": parent["gain"] if parent else 1.,
                                   "muted": parent["muted"] if parent else False,
                                   "peak": float(np.max(np.abs(audio))),
                                   "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                                   "parent_id": parent_id})
            data["history"].append({"tracks": data["tracks"], "notes": data["notes"]})
            if parent:
                index = data["tracks"].index(parent)
                data["tracks"] = data["tracks"][:index] + new_tracks + data["tracks"][index + 1:]
            else:
                data["tracks"] = new_tracks
            data["notes"] = notes or ["名称为算法候选，请逐轨独听、命名后选择静音。"]
            data["revision"] += 1
            self.save(data)
            return data

    def update(self, identifier: str, values: list[dict], name: str | None = None) -> dict:
        with self.lock:
            data = self.load(identifier)
            data["tracks"] = validate_controls(data["tracks"], values, subset=False)
            if name is not None:
                data["name"] = safe_name(name)
            self.save(data)
            return data

    def undo(self, identifier: str) -> dict:
        with self.lock:
            data = self.load(identifier)
            if not data["history"]:
                raise ProcessingError("没有可以撤销的分离操作。")
            snapshot = data["history"].pop()
            data["tracks"], data["notes"] = snapshot["tracks"], snapshot["notes"]
            data["revision"] += 1
            self.save(data)
            return data

    def mix(self, data: dict, values: list[dict]) -> np.ndarray:
        tracks = validate_controls(data["tracks"], values, subset=True)
        result = np.zeros((data["samples"], data["channels"]), dtype=np.float64)
        for track in tracks:
            if not track["muted"] and track["gain"]:
                result += self.track_audio(data, track).astype(np.float64) * track["gain"]
        return result.astype(np.float32)

    def export(self, identifier: str, values: list[dict], format: str, mode: str) -> dict:
        if format not in ("wav", "flac", "mp3") or mode not in ("mix", "stems"):
            raise ProcessingError("不支持的导出格式。")
        data = self.load(identifier)
        tracks = validate_controls(data["tracks"], values, subset=True)
        out_dir = self.root / "exports"
        out_dir.mkdir(exist_ok=True)
        token = uuid.uuid4().hex
        peak, scale = 0., 1.
        if mode == "stems":
            extension, label = ".zip", "独立音轨"
            destination = out_dir / (token + extension)
            with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as bundle:
                for i, track in enumerate(tracks, 1):
                    bundle.write(self.directory(identifier) / track["file"],
                                 f'{i:02d}_{safe_name(track["name"])}.wav')
                bundle.writestr("混音设置.json", json.dumps({
                    "sample_rate": data["sample_rate"], "tracks": tracks,
                    "note": "音轨是未应用音量/静音的32位浮点WAV；混音设置单独保留。",
                }, ensure_ascii=False, indent=2))
            message = "已导出独立浮点 WAV 音轨和混音设置；每轨保留相对增益。"
        else:
            audio = self.mix(data, tracks)
            peak = float(np.max(np.abs(audio)))
            ceiling = 10 ** ((-1.0 if format == "mp3" else -0.1) / 20)
            scale = min(1., ceiling / max(peak, 1e-12))
            audio *= scale
            extension, label = "." + format, "混音"
            destination = out_dir / (token + extension)
            codec = {"wav": ["-c:a", "pcm_s24le"], "flac": ["-c:a", "flac", "-sample_fmt", "s32"],
                     "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"]}[format]
            args = [_program("ffmpeg"), "-v", "error", "-n", "-f", "f32le", "-ar",
                    str(data["sample_rate"]), "-ac", str(data["channels"]), "-i", "pipe:0", *codec, str(destination)]
            result = subprocess.run(args, input=audio.astype("<f4").tobytes(), capture_output=True,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise ProcessingError("导出失败：" + result.stderr.decode("utf-8", errors="replace")[-1200:])
            message = "已导出所选轨道的混音。"
            if scale < 1:
                message += f"为防止削波，整体降低了 {-20 * math.log10(scale):.2f} dB；轨间比例保留。"
        filename = f'{safe_name(data["name"])}_{label}{extension}'
        atomic_json(out_dir / (token + ".download.json"), {"filename": filename})
        return {"url": f"/download/{token}{extension}",
                "filename": filename,
                "peak": peak, "scale": scale, "message": message, "path": str(destination)}
