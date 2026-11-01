"""Validated, cancellable process boundary for the official six-source Demucs model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "demucs"
CHECKPOINT = MODEL_DIR / "5c90dfd2-34c22ccb.th"
MODEL_SHA256 = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")


class SeparationError(RuntimeError):
    pass


def check_model() -> dict:
    """Read-only availability report; loading never silently downloads models."""
    missing = []
    if not CHECKPOINT.is_file():
        missing.append("Demucs 六轨权重")
    for module in ("demucs", "julius", "openunmix", "dora", "torchaudio"):
        if not (ROOT / ".deps" / module).is_dir():
            import importlib.util
            if importlib.util.find_spec(module) is None:
                missing.append(module)
    return {"available": not missing, "model": "htdemucs_6s", "missing": missing,
            "license": "MIT", "device": "cpu", "sources": list(SOURCES)}


def _validate_checkpoint():
    if not CHECKPOINT.is_file():
        raise SeparationError("缺少 Demucs 六轨权重，请运行 python tools/setup_demucs.py。")
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise SeparationError("Demucs 模型校验失败，停止加载。请从官方来源重新安装。")


def separate_demucs(audio: np.ndarray, sr: int,
                    progress: Callable[[float, str], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> dict[str, np.ndarray]:
    """Return six aligned float32 stems, without per-stem normalization.

    Model labels are candidates, not guarantees about the content. The network
    does not guarantee that its six outputs add exactly to the mixture; callers
    should keep an explicit residual when an exactly reversible mixer is wanted.
    """
    say = progress or (lambda _p, _m: None)
    is_cancelled = cancelled or (lambda: False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or audio.shape[0] == 0:
        raise ValueError("音频必须是非空单声道或立体声 [采样, 声道] 数组。")
    if not isinstance(sr, (int, np.integer)) or not 8000 <= sr <= 192000:
        raise ValueError("采样率应为 8000–192000 Hz。")
    if not np.isfinite(audio).all():
        raise ValueError("输入音频包含无效数值。")
    if is_cancelled():
        raise InterruptedError("已取消音轨分离。")
    _validate_checkpoint()
    say(1, "加载 Demucs 六轨神经网络…")
    with tempfile.TemporaryDirectory(prefix="stemstudio-demucs-") as folder:
        temporary = Path(folder)
        source = temporary / "input.npy"
        np.save(source, audio, allow_pickle=False)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / ".deps"), str(ROOT), env.get("PYTHONPATH", "")])
        env["PYTHONUTF8"] = "1"
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "1"
        try:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("demucs_worker.py")),
                 str(source), str(int(sr)), str(temporary)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise SeparationError(f"无法启动 Demucs 推理：{exc}") from exc
        lines: queue.Queue[str | None] = queue.Queue()
        assert process.stdout is not None

        def read_output():
            try:
                with process.stdout:
                    for line in process.stdout:
                        lines.put(line.rstrip())
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        last_messages: list[str] = []
        ended = False
        try:
            while not ended:
                if is_cancelled():
                    raise InterruptedError("已取消音轨分离。")
                try:
                    line = lines.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    ended = True
                elif line.startswith("PROGRESS\t"):
                    try:
                        _, value, message = line.split("\t", 2)
                        say(float(value), message)
                    except (ValueError, TypeError):
                        last_messages.append(line)
                elif line:
                    last_messages.append(line)
                    last_messages = last_messages[-12:]
            code = process.wait()
            if code != 0:
                detail = "；".join(last_messages[-5:]) or f"退出码 {code}"
                raise SeparationError(f"Demucs 分离失败：{detail}")
            info = json.loads((temporary / "result.json").read_text(encoding="utf-8"))
            if info.get("sources") != list(SOURCES):
                raise SeparationError("Demucs 返回了不符合预期的音轨列表。")
            outputs = {}
            for name in SOURCES:
                output = np.load(temporary / f"{name}.npy", allow_pickle=False)
                if output.shape != audio.shape or not np.isfinite(output).all():
                    raise SeparationError(f"{name} 音轨的数据或长度无效。")
                outputs[name] = output.astype(np.float32, copy=False)
            say(100, "六条神经网络音轨已生成。")
            return outputs
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join(timeout=5)
