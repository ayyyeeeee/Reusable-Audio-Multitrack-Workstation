"""Private Torch-only worker. No SciPy import; prevents Windows OpenMP clashes."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
import torch
import julius
from demucs.htdemucs import HTDemucs
from demucs.apply import TensorChunk, apply_model

CHECKPOINT = ROOT / "models" / "demucs" / "5c90dfd2-34c22ccb.th"
CHECKSUM = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
SOURCES = ["drums", "bass", "other", "vocals", "guitar", "piano"]


def progress(value, message):
    print(f"PROGRESS\t{value:.2f}\t{message}", flush=True)


def main():
    source, rate, destination = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    # Check the entire known official checkpoint before deserializing. Restricted
    # unpickling permits only the official model class and Fraction metadata.
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKSUM:
        raise RuntimeError("Demucs checkpoint checksum mismatch")
    with torch.serialization.safe_globals([HTDemucs, Fraction]):
        package = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if package["klass"] is not HTDemucs:
        raise RuntimeError("Unexpected model class")
    model = HTDemucs(*package["args"], **package["kwargs"])
    model.load_state_dict(package["state"], strict=True)
    del package
    model.eval()
    if list(model.sources) != SOURCES:
        raise RuntimeError("Unexpected six-source order")
    original = np.load(source, allow_pickle=False)
    length, channels = original.shape
    audio = torch.from_numpy(original.T.copy())
    if channels == 1:
        audio = audio.repeat(2, 1)
    audio = julius.resample_frac(audio, rate, model.samplerate)
    reference = audio.mean(dim=0)
    mean = audio.mean()
    # The customary mono-reference normalisation fails for opposite-polarity
    # stereo: a loud recording can have a silent L+R mean. Keep the customary
    # scale for ordinary stereo while bounding it with the actual channel energy.
    channel_scale = torch.sqrt(torch.mean((audio - mean) ** 2))
    std = torch.maximum(reference.std(unbiased=False), channel_scale * .25).clamp_min(1e-8)
    normalized = (audio - mean) / std
    segment_seconds = float(model.segment)
    segment = int(segment_seconds * model.samplerate)
    stride = int(segment * .75)
    starts = list(range(0, normalized.shape[-1], stride))
    result = torch.zeros(len(SOURCES), 2, normalized.shape[-1])
    weights = torch.zeros(normalized.shape[-1])
    window = torch.cat([torch.arange(1, segment // 2 + 1),
                        torch.arange(segment - segment // 2, 0, -1)]).float()
    window /= window.max()
    progress(5, f"CPU 推理：{len(starts)} 段，六条候选音轨")
    with torch.inference_mode():
        for index, start in enumerate(starts):
            chunk = TensorChunk(normalized[None], start, segment)
            estimate = apply_model(model, chunk, shifts=0, split=False,
                                   device="cpu", segment=segment_seconds)[0]
            count = estimate.shape[-1]
            result[..., start:start + count] += estimate * window[:count]
            weights[start:start + count] += window[:count]
            progress(5 + 85 * (index + 1) / len(starts),
                     f"神经网络分离：{index + 1}/{len(starts)} 段")
        result /= weights.clamp_min(1e-8)
        # Use a shared scale for every source, never independently normalize.
        # Split the recording DC offset equally to avoid adding it six times.
        result = result * std + mean / len(SOURCES)
        if channels == 1:
            result = result.mean(dim=1, keepdim=True)
        for index, name in enumerate(SOURCES):
            output = julius.resample_frac(result[index], model.samplerate, rate)
            if output.shape[-1] < length:
                output = torch.nn.functional.pad(output, (0, length - output.shape[-1]))
            np.save(destination / f"{name}.npy", output[..., :length].T.contiguous().numpy(), allow_pickle=False)
            progress(90 + 9 * (index + 1) / len(SOURCES), f"保存 {name} 候选轨")
    (destination / "result.json").write_text(json.dumps({"sources": SOURCES,
         "model": "htdemucs_6s", "rate": rate, "samples": length}), encoding="utf-8")


if __name__ == "__main__":
    main()
