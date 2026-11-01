"""保守的重复瞬态模板学习与减除；不需要预训练神经网络。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable

import numpy as np
from scipy import ndimage, signal


class ProcessingError(Exception):
    """可向使用者展示的处理错误。"""


@dataclass
class ProcessConfig:
    strength: float = 0.65
    min_bpm: float = 50.0
    max_bpm: float = 200.0
    reference_path: str | None = None
    export_timestamps: bool = True
    method: str = "classic"


def _validate_config(config: ProcessConfig) -> None:
    if config.method not in ("classic", "neural"):
        raise ProcessingError("处理模式必须是 neural（神经网络）或 classic（传统模板）。")
    if not (math.isfinite(config.strength) and 0 <= config.strength <= 1):
        raise ProcessingError("处理力度必须为 0 到 1 之间的数字。")
    if not (math.isfinite(config.min_bpm) and math.isfinite(config.max_bpm)
            and 0 < config.min_bpm < config.max_bpm <= 600):
        raise ProcessingError("速度范围无效：最小值必须大于 0 且小于最大值，最大值不超过 600。")


def _program(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ProcessingError(f"找不到 {name}。请安装 FFmpeg 并将 ffmpeg 和 ffprobe 加入 PATH。")
    return path


def _decode(path: Path, *, max_duration: float = 1800) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise ProcessingError(f"音频文件不存在：{path}")
    try:
        probe = subprocess.run(
            [_program("ffprobe"), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels:format=duration", "-of", "json", str(path)],
            capture_output=True, check=True,
        )
        info = json.loads(probe.stdout)
        stream = info["streams"][0]
        sr, channels = int(stream["sample_rate"]), int(stream["channels"])
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise ProcessingError(f"无法读取音频信息：{path.name}") from exc
    if channels not in (1, 2):
        raise ProcessingError("仅支持单声道或立体声输入。")
    if sr < 8000 or sr > 192000:
        raise ProcessingError("支持 8 kHz 至 192 kHz 的采样率。")
    # Full-recording analysis is intentionally bounded to prevent exhausting memory.
    try:
        duration = float(info.get("format", {}).get("duration") or 0)
    except (ValueError, TypeError):
        duration = 0
    limit_message = ("参考音不能超过 30 秒。" if max_duration == 30 else
                     "录音过长或采样数据过大；当前版本最多 30 分钟且解码后不超过 300 MiB，请先分段。")
    if duration > max_duration or (duration and duration * sr * channels * 4 > 300 * 1024 * 1024):
        raise ProcessingError(limit_message)
    try:
        decoded = subprocess.run(
            [_program("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0",
             "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ProcessingError(f"音频解码失败：{detail or path.name}") from exc
    audio = np.frombuffer(decoded.stdout, dtype="<f4")
    if not len(audio) or len(audio) % channels:
        raise ProcessingError("音频为空或解码结果不完整。")
    audio = audio.reshape(-1, channels).copy()
    if len(audio) / sr > max_duration or audio.nbytes > 300 * 1024 * 1024:
        raise ProcessingError(limit_message)
    if not np.isfinite(audio).all():
        raise ProcessingError("音频包含无效采样值，无法处理。")
    return audio, sr


def _resample(audio: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    if old_sr == new_sr:
        return audio
    ratio = Fraction(new_sr, old_sr)
    return signal.resample_poly(audio, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def _highpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    cutoff = min(cutoff, sr * 0.39)
    sos = signal.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _analysis_signal(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int, int]:
    rate = min(sr, 16000)
    reduced = _resample(audio, sr, rate)
    high = _highpass(reduced, rate, min(3000, rate * .30))
    # Do not average L/R: opposite-polarity signals could cancel.
    channel = int(np.argmax(np.mean(high.astype(np.float64) ** 2, axis=0)))
    return high[:, channel], rate, channel


def _candidate_peaks(y: np.ndarray, sr: int) -> np.ndarray:
    if len(y) < sr // 10:
        return np.array([], dtype=int)
    power = ndimage.uniform_filter1d(y.astype(np.float64) ** 2, size=max(3, round(sr * .004)))
    hop = max(1, round(sr * .002))
    envelope = np.sqrt(np.maximum(0, power[::hop]))
    maximum = float(np.max(envelope))
    if maximum < 1e-8:
        return np.array([], dtype=int)
    prominence = max(float(np.median(envelope)) * .45, maximum * .02, 1e-10)
    peaks, _ = signal.find_peaks(envelope, distance=max(1, round(.10 * sr / hop)), prominence=prominence)
    return (peaks * hop).astype(int)


def _local_match(y: np.ndarray, query: np.ndarray, center: int, pre: int,
                 radius: int) -> tuple[float, int]:
    """归一化互相关；返回最佳得分和原信号中的模板起点。"""
    start = max(0, center - pre - radius)
    stop = min(len(y), center - pre + radius + len(query) + 1)
    sample = y[start:stop]
    if len(sample) < len(query):
        return 0.0, start
    corr = signal.correlate(sample, query, mode="valid", method="direct")
    sq = np.r_[0.0, np.cumsum(sample.astype(np.float64) ** 2)]
    energy = sq[len(query):] - sq[:-len(query)]
    denom = np.sqrt(np.maximum(energy, 0) * float(np.dot(query, query))) + 1e-12
    scores = corr / denom
    best = int(np.argmax(scores))
    return float(scores[best]), start + best


def _tempo_consistency(times: np.ndarray, min_bpm: float, max_bpm: float) -> tuple[float, float | None]:
    if len(times) < 3:
        return 0.0, None
    gaps = np.diff(np.sort(times))
    lo, hi = 60 / max_bpm, 60 / min_bpm
    # Limited samples prevent quadratic growth for long recordings.
    chosen = gaps[np.linspace(0, len(gaps) - 1, min(len(gaps), 100)).astype(int)]
    periods = [g / k for g in chosen for k in range(1, 7) if lo <= g / k <= hi]
    if not periods:
        return 0.0, None
    best_score, best_period = 0.0, None
    for period in periods:
        beats = np.maximum(1, np.rint(gaps / period))
        tolerance = min(.060, max(.016, period * .055))
        ok = (beats <= 8) & (np.abs(gaps - beats * period) <= tolerance)
        score = float(np.mean(ok))
        if score > best_score:
            best_score, best_period = score, period
    return best_score, (60 / best_period if best_period else None)


def _auto_query(y: np.ndarray, sr: int, config: ProcessConfig) -> tuple[np.ndarray, int, float | None]:
    peaks = _candidate_peaks(y, sr)
    if len(peaks) < 5:
        raise ProcessingError("重复瞬态不足，无法可靠学习节拍器；可提供只含节拍器的参考音。")
    # A large recording may have thousands of onsets: learn from a spread-out subset.
    if len(peaks) > 400:
        peaks = peaks[np.linspace(0, len(peaks) - 1, 400).astype(int)]
    pre, post = round(sr * .006), round(sr * .008)
    valid = peaks[(peaks >= pre) & (peaks + post < len(y))]
    if len(valid) < 5:
        raise ProcessingError("可用于学习的瞬态不足，请提供节拍器参考音。")
    strengths = np.array([np.dot(y[p-pre:p+post], y[p-pre:p+post]) for p in valid])
    top = np.argsort(strengths)[-min(20, len(valid)):]
    spread = np.linspace(0, len(valid) - 1, min(16, len(valid))).astype(int)
    choices = np.unique(np.r_[top, spread])
    best, best_query, best_bpm = -1.0, None, None
    for index in choices:
        seed = int(valid[index])
        query = y[seed-pre:seed+post].copy()
        if np.dot(query, query) < 1e-12:
            continue
        times, scores = [], []
        for p in valid:
            corr, start = _local_match(y, query, int(p), pre, round(.012 * sr))
            if corr >= .60:
                times.append((start + pre) / sr)
                scores.append(corr)
        if len(times) < 5:
            continue
        consistency, bpm = _tempo_consistency(np.array(times), config.min_bpm, config.max_bpm)
        if consistency < .55:
            continue
        merit = len(times) * consistency * float(np.median(scores))
        if merit > best:
            best, best_query, best_bpm = merit, query, bpm
    if best_query is None:
        raise ProcessingError("未发现足够相似且节奏相容的重复瞬态；请调整速度范围或提供纯节拍器参考音。")
    return best_query, pre, best_bpm


def _reference_query(audio: np.ndarray, sr: int, target_rate: int) -> tuple[np.ndarray, int]:
    if len(audio) / sr > 30:
        raise ProcessingError("参考音应不超过 30 秒，且只包含节拍器声。")
    reduced = _resample(audio, sr, target_rate)
    high = _highpass(reduced, target_rate, min(3000, target_rate * .30))
    channel = int(np.argmax(np.mean(high.astype(np.float64) ** 2, axis=0)))
    y = high[:, channel]
    power = ndimage.uniform_filter1d(y.astype(np.float64) ** 2, size=max(3, round(target_rate * .004)))
    anchor = int(np.argmax(power))
    pre, post = round(target_rate * .006), round(target_rate * .008)
    if anchor < pre or anchor + post >= len(y):
        raise ProcessingError("参考音中的节拍声离文件边缘太近，请在前后保留少量空白。")
    query = y[anchor-pre:anchor+post].copy()
    if np.dot(query, query) < 1e-12:
        raise ProcessingError("参考音中没有足够清晰的高频节拍器瞬态。")
    return query, pre


def _find_matches(y: np.ndarray, sr: int, query: np.ndarray, pre: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < len(query):
        return np.array([], dtype=int), np.array([])
    corr = signal.fftconvolve(y, query[::-1], mode="valid")
    squared = np.r_[0., np.cumsum(y.astype(np.float64) ** 2)]
    local_energy = squared[len(query):] - squared[:-len(query)]
    denominator = np.sqrt(np.maximum(local_energy, 0) * float(np.dot(query, query))) + 1e-12
    scores = corr / denominator
    starts, props = signal.find_peaks(scores, height=.60, distance=max(1, round(sr * .10)))
    return (starts + pre).astype(int), props["peak_heights"]


def _window_fade(length: int, sr: int) -> np.ndarray:
    fade = min(round(sr * .005), length // 2)
    result = np.ones(length, dtype=np.float32)
    if fade:
        curve = np.sin(np.linspace(0, np.pi / 2, fade, endpoint=True)) ** 2
        result[:fade], result[-fade:] = curve, curve[::-1]
    return result


def _learn_template(audio: np.ndarray, sr: int, centers: np.ndarray, channel: int) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """对齐多拍后用中位数及频率相干性学习重复声。"""
    before, after = round(.015 * sr), round(.195 * sr)
    length = before + after
    usable = centers[(centers >= before) & (centers + after < len(audio))]
    if len(usable) < 4:
        raise ProcessingError("完整且可信的节拍器事件少于 4 个，无法从混音安全学习模板。")
    filtered = _highpass(audio, sr, 850)
    reference = _highpass(audio[:, channel:channel+1], sr, min(3000, sr * .30))[:, 0]
    qpre, qpost = round(.006 * sr), round(.008 * sr)
    seed = int(usable[len(usable) // 2])
    query = reference[seed-qpre:seed+qpost]
    aligned, gains = [], []
    qenergy = float(np.dot(query, query)) + 1e-12
    refined_centers = centers.copy()
    good = np.zeros(len(centers), dtype=bool)
    for i, center in enumerate(centers):
        score, start = _local_match(reference, query, int(center), qpre, max(2, round(.002 * sr)))
        refined = start + qpre
        if score >= .48:
            refined_centers[i] = refined
            good[i] = True
    usable = refined_centers[good & (refined_centers >= before) & (refined_centers + after < len(audio))]
    if len(usable) > 64:
        usable = usable[np.linspace(0, len(usable) - 1, 64).astype(int)]
    for refined in usable:
        start = int(refined) - qpre
        match = reference[start:start+len(query)]
        if len(match) != len(query):
            continue
        gain = float(np.dot(match, query) / qenergy)
        if gain <= .05:
            continue
        aligned.append(filtered[refined-before:refined+after])
        gains.append(gain)
    if len(aligned) < 4:
        raise ProcessingError("节拍器事件不够相似，无法可靠建立模板；请试用纯节拍器参考音。")
    gains = np.asarray(gains)
    median_gain = float(np.median(gains))
    stack = np.stack([w / max(g, median_gain * .25) for w, g in zip(aligned, gains)], axis=0)
    median = np.median(stack, axis=0)
    fade = _window_fade(length, sr)
    freq = np.fft.rfftfreq(length, 1 / sr)
    result = np.empty_like(median, dtype=np.float32)
    for ch in range(audio.shape[1]):
        spectra = np.fft.rfft(stack[:, :, ch] * fade, axis=1)
        coherence = np.abs(np.mean(spectra, axis=0)) / (np.sqrt(np.mean(np.abs(spectra) ** 2, axis=0)) + 1e-12)
        coherence = ndimage.gaussian_filter1d(coherence, max(1.0, 35 / (sr / length)))
        reliability = np.clip((coherence - .18) / .57, 0, 1)
        # Leave the low frequencies alone; soften the transition around 1 kHz.
        reliability *= np.clip((freq - 650) / 600, 0, 1)
        estimate = np.fft.irfft(np.fft.rfft(median[:, ch] * fade) * reliability, n=length)
        result[:, ch] = (estimate * fade).astype(np.float32)
    return result, before, refined_centers, good


def _subtract(audio: np.ndarray, sr: int, centers: np.ndarray, scores: np.ndarray,
              template: np.ndarray, before: int, strength: float,
              *, fit_template: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    result = audio.copy()
    length = len(template)
    # Fit per-event gain over the high-frequency 14 ms attack, with one gain for stereo.
    high_audio = _highpass(audio, sr, min(3000, sr * .30))
    fit_source = template if fit_template is None else fit_template
    if fit_source.shape != template.shape:
        raise ProcessingError("拟合模板与减除模板的形状不一致。")
    high_template = _highpass(fit_source, sr, min(3000, sr * .30))
    left = max(0, before - round(.006 * sr))
    right = min(length, before + round(.008 * sr))
    model = high_template[left:right]
    energy = float(np.sum(model.astype(np.float64) ** 2)) + 1e-12
    raw = []
    for center in centers:
        start = int(center) - before
        if start + left < 0 or start + right > len(audio):
            raw.append(0.0)
            continue
        target = high_audio[start+left:start+right]
        raw.append(max(0., float(np.sum(target.astype(np.float64) * model) / energy)))
    positive = np.asarray([v for v in raw if v > 0])
    middle = float(np.median(positive)) if len(positive) else 0
    events = []
    for center, score, gain in zip(centers, scores, raw):
        start = int(center) - before
        # Unusually large fits can be caused by overlapping piano attacks.
        gain = min(gain, middle * 2.0) if middle else 0.0
        template_start = max(0, -start)
        template_end = min(length, len(audio) - start)
        processed = bool(gain > 0 and template_end > template_start and strength > 0)
        if processed:
            result[start+template_start:start+template_end] -= strength * gain * template[template_start:template_end]
        events.append({"time_seconds": int(center) / sr, "similarity": float(score),
                       "relative_gain": gain, "processed": processed})
    return result, events


def remove_samples(audio: np.ndarray, sr: int, config: ProcessConfig,
                   reference: tuple[np.ndarray, int] | None = None,
                   progress: Callable[[float, str], None] | None = None) -> tuple[np.ndarray, dict]:
    """处理 float 音频数组；无可靠匹配时抛出 ProcessingError。"""
    _validate_config(config)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or len(audio) < sr:
        raise ProcessingError("需要至少 1 秒的单声道或立体声音频。")
    if sr < 8000 or sr > 192000 or not np.isfinite(audio).all():
        raise ProcessingError("音频采样率或数据无效。")
    say = progress or (lambda _p, _m: None)
    if config.method == "neural":
        from .neural import estimate_piano_isolated
        from .multiclass import remove_guided
        piano = estimate_piano_isolated(audio, sr, progress=say)
        return remove_guided(audio, sr, config, piano, reference=reference, progress=say)
    say(15, "分析录音中的瞬态与重复模式…")
    y, analysis_sr, channel = _analysis_signal(audio, sr)
    bpm = None
    if reference is None:
        query, pre, bpm = _auto_query(y, analysis_sr, config)
    else:
        ref, ref_sr = reference
        query, pre = _reference_query(np.asarray(ref, dtype=np.float32), ref_sr, analysis_sr)
    say(42, "识别相似的节拍声…")
    matches, scores = _find_matches(y, analysis_sr, query, pre)
    if len(matches) < 5:
        raise ProcessingError("匹配到的节拍声不足 5 次；未生成输出。请检查参考音与原录音的设备和环境是否一致。")
    if reference is None:
        consistency, _ = _tempo_consistency(matches / analysis_sr, config.min_bpm, config.max_bpm)
        if consistency < .50:
            raise ProcessingError("检测结果的节奏一致性不足，未生成输出；可提供节拍器参考音。")
    centers = np.rint(matches.astype(np.float64) * sr / analysis_sr).astype(int)
    say(60, "学习重复节拍模板并保守减除…")
    template, before, centers, good = _learn_template(audio, sr, centers, channel)
    cleaned, events = _subtract(audio, sr, centers[good], scores[good], template, before, config.strength)
    count = sum(event["processed"] for event in events)
    warnings = ["模板来自录音中的重复声，可能漏掉不同音色的重拍；请试听并检查钢琴起音。",
                "该算法无法保证完全分离，且不是预训练神经网络。"]
    if count == 0:
        warnings.append("没有减除任何事件；如果力度设为 0，输出将与输入相同。")
    return cleaned, {"processed_count": count, "candidate_count": len(events),
                     "estimated_bpm": bpm, "events": events, "warnings": warnings}


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> bool:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scaled = peak > .999999
    if scaled:
        audio = audio * (.999999 / peak)
    # Render to a unique temporary file, then use exclusive creation for final output.
    fd, temp_name = tempfile.mkstemp(suffix=".wav", dir=str(path.parent), prefix=".metronome-")
    os.close(fd)
    try:
        proc = subprocess.Popen(
            [_program("ffmpeg"), "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
             "-ac", str(audio.shape[1]), "-i", "-", "-c:a", "pcm_s24le", temp_name],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _, stderr = proc.communicate(np.asarray(audio, dtype="<f4").tobytes())
        if proc.returncode:
            raise ProcessingError("写入 WAV 失败：" + stderr.decode("utf-8", errors="replace"))
        with open(temp_name, "rb") as src, open(path, "xb") as dst:
            shutil.copyfileobj(src, dst)
    finally:
        try:
            os.remove(temp_name)
        except FileNotFoundError:
            pass
    return scaled


def process_file(input_path: str, output_path: str, *, config: ProcessConfig | None = None,
                 progress: Callable[[float, str], None] | None = None) -> dict:
    config = config or ProcessConfig()
    _validate_config(config)
    source, output = Path(input_path).expanduser().resolve(), Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise ProcessingError("输入音频不存在。")
    if output.suffix.lower() != ".wav" or not output.parent.is_dir():
        raise ProcessingError("输出目录须存在，且输出文件名须以 .wav 结尾。")
    if source == output or output.exists():
        raise ProcessingError("输出文件已存在或与输入相同；请选择新文件名。")
    csv_path = output.with_name(output.stem + "_节拍事件.csv") if config.export_timestamps else None
    if csv_path and csv_path.exists():
        raise ProcessingError("时间戳文件已存在；为避免覆盖，请更换输出文件名。")
    say = progress or (lambda _p, _m: None)
    say(2, "读取输入录音…")
    audio, sr = _decode(source)
    reference = _decode(Path(config.reference_path).expanduser().resolve(), max_duration=30) if config.reference_path else None
    cleaned, report = remove_samples(audio, sr, config, reference=reference, progress=say)
    say(90, "保存 WAV 和时间戳…")
    if _write_wav(output, cleaned, sr):
        report["warnings"].append("为避免 PCM 裁剪，输出整体音量作了轻微衰减。")
    if csv_path:
        try:
            with open(csv_path, "x", encoding="utf-8-sig", newline="") as f:
                fields = ["time_seconds", "similarity", "relative_gain", "processed"]
                extras = sorted({key for event in report["events"] for key in event if key not in fields})
                writer = csv.DictWriter(f, fieldnames=fields + extras)
                writer.writeheader()
                writer.writerows(report["events"])
        except OSError as exc:
            report["warnings"].append(f"音频已保存，但时间戳未能写入：{exc}")
            csv_path = None
    report.update({"event_csv": str(csv_path) if csv_path else None, "sample_rate": sr,
                   "channels": audio.shape[1], "duration_seconds": len(audio) / sr})
    say(100, "完成。请试听输出并检查与钢琴起音重合的部分。")
    return report
