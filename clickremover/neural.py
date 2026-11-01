"""Use an attributed pretrained BS-RoFormer piano model locally on short chunks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

import numpy as np

from .engine import ProcessingError, _resample


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "piano-separation-small"
DEPS_DIR = Path(__file__).resolve().parent.parent / ".deps"
MODEL_SHA256 = "c3c6666071d3de5106fd6ce2bf7cfa93ccab48e592d00504de1d1ffe51bfb211"
CODE_SHA256 = "c5675e717246b74b3e8c76088960f242759cbbc791fe58c4fd29ac6dee03c902"
CONFIG_SHA256 = "0624d982534169bfe300312f4b9378800e02259b8c0d27809878f970ac71a072"
MODEL_RATE = 44100


def _load_model():
    checkpoint = MODEL_DIR / "model.safetensors"
    code = MODEL_DIR / "inference.py"
    config_file = MODEL_DIR / "config.json"
    if not all(path.is_file() for path in (checkpoint, code, config_file)):
        raise ProcessingError("缺少神经网络模型文件；请按使用说明安装 piano-separation-small 权重。")
    for path, expected in ((checkpoint, MODEL_SHA256), (code, CODE_SHA256), (config_file, CONFIG_SHA256)):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ProcessingError(f"神经网络文件校验失败：{path.name}；请从使用说明中的官方来源重新下载。")
    if DEPS_DIR.is_dir() and str(DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(DEPS_DIR))
    try:
        import torch
        from safetensors.torch import load_file
        # Upstream inference code is included under its original license. Importing it
        # does not invoke its CLI or automatic download function.
        spec = importlib.util.spec_from_file_location("piano_separator_upstream", code)
        if spec is None or spec.loader is None:
            raise RuntimeError("模型代码无法加载")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError) as exc:
        raise ProcessingError(f"神经网络依赖缺失或无法加载：{exc}。请查看 README 中的安装步骤。") from exc
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
        model = module.BSRoformer(**config).eval()
        state = load_file(str(checkpoint), device="cpu")
        model.load_state_dict({key[7:] if key.startswith("module.") else key: value
                               for key, value in state.items()})
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        raise ProcessingError(f"无法加载神经网络模型：{exc}") from exc
    return torch, model


def estimate_piano(audio: np.ndarray, sr: int,
                   progress: Callable[[float, str], None] | None = None) -> np.ndarray:
    """Return the model's piano estimate at the input length, rate and channel count.

    Short overlapping chunks keep the time-attention memory requirements practical on CPU.
    This is a guide to local subtraction, not a replacement for the source recording.
    """
    say = progress or (lambda _p, _m: None)
    say(8, "加载预训练钢琴神经网络…")
    torch, model = _load_model()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    channels = audio.shape[1]
    reduced = _resample(audio, sr, MODEL_RATE)
    if channels == 1:
        reduced = np.repeat(reduced, 2, axis=1)
    mono = np.mean(reduced, axis=1)
    mean = float(np.mean(mono))
    std = max(float(np.std(mono)), 1e-8)
    normalized = (reduced - mean) / std
    segment = int(3.5 * MODEL_RATE)
    stride = int(segment * .75)
    starts = list(range(0, len(normalized), stride))
    result = np.zeros((len(normalized), 2), dtype=np.float32)
    weights = np.zeros(len(normalized), dtype=np.float32)
    window = np.maximum(np.hanning(segment).astype(np.float32), .001)
    with torch.inference_mode():
        for index, start in enumerate(starts):
            chunk = normalized[start:start+segment]
            actual = len(chunk)
            if actual < segment:
                # Edge padding handles short final chunks and sub-second files.
                chunk = np.pad(chunk, ((0, segment - actual), (0, 0)), mode="edge")
            tensor = torch.from_numpy(chunk.T.copy())[None]
            try:
                estimate = model(tensor)[0, 0].cpu().numpy().T
            except RuntimeError as exc:
                raise ProcessingError(f"神经网络推理失败：{exc}") from exc
            result[start:start+actual] += estimate[:actual] * window[:actual, None]
            weights[start:start+actual] += window[:actual]
            say(10 + 60 * (index + 1) / len(starts),
                f"神经网络分离中（{index + 1}/{len(starts)} 段）…")
    result = result / np.maximum(weights[:, None], 1e-8)
    result = result * std + mean
    if channels == 1:
        result = np.mean(result, axis=1, keepdims=True)
    restored = _resample(result.astype(np.float32), MODEL_RATE, sr)
    if len(restored) < len(audio):
        restored = np.pad(restored, ((0, len(audio) - len(restored)), (0, 0)), mode="edge")
    return restored[:len(audio)].astype(np.float32)


def estimate_piano_isolated(audio: np.ndarray, sr: int,
                            progress: Callable[[float, str], None] | None = None) -> np.ndarray:
    """Run inference in a child process to isolate conflicting OpenMP runtimes.

    Only temporary local arrays are passed to the worker; no audio is uploaded.
    A worker failure is reported explicitly and never triggers a classic fallback.
    """
    say = progress or (lambda _p, _m: None)
    with tempfile.TemporaryDirectory(prefix="piano-neural-") as temporary:
        directory = Path(temporary)
        source, destination = directory / "source.npy", directory / "estimate.npy"
        np.save(source, np.asarray(audio, dtype=np.float32), allow_pickle=False)
        env = os.environ.copy()
        project = str(Path(__file__).resolve().parent.parent)
        entries = [project, str(DEPS_DIR)]
        if env.get("PYTHONPATH"):
            entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(entries)
        env["PYTHONUTF8"] = "1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "clickremover.neural_worker", str(source), str(sr), str(destination)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", env=env, creationflags=flags,
            )
        except OSError as exc:
            raise ProcessingError(f"无法启动神经网络推理进程：{exc}") from exc
        messages: list[str] = []
        assert proc.stdout is not None
        with proc.stdout:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line.startswith("PROGRESS\t"):
                    try:
                        _, value, message = line.split("\t", 2)
                        say(float(value), message)
                    except (ValueError, TypeError):
                        continue
                elif line:
                    messages.append(line)
                    messages = messages[-8:]
        code = proc.wait()
        if code != 0 or not destination.is_file():
            detail = "；".join(messages[-3:]) or f"进程退出码 {code}"
            raise ProcessingError(f"神经网络推理失败：{detail}")
        try:
            estimate = np.load(destination, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProcessingError(f"无法读取神经网络输出：{exc}") from exc
    if estimate.shape != audio.shape or not np.isfinite(estimate).all():
        raise ProcessingError("神经网络输出的形状或数据无效。")
    return estimate.astype(np.float32, copy=False)
