"""Learn audible candidate stems without claiming to identify every sound source.

NMF learns nonnegative spectral atoms from this recording. Atoms are grouped by
their time envelopes and spectra, then shared stereo masks recover candidate
waveforms. This is a blind decomposition, not an instrument classifier; one
instrument can span groups and one group can still contain several instruments.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import numpy as np
from scipy import ndimage, signal

from clickremover.engine import ProcessConfig, ProcessingError

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def _check_cancel(cancelled: Cancelled | None) -> None:
    if cancelled is not None and cancelled():
        raise ProcessingError("已取消音轨拆分。")


def _validate(audio: np.ndarray, sr: int) -> np.ndarray:
    if not isinstance(sr, (int, np.integer)) or isinstance(sr, bool) or not 8000 <= sr <= 192000:
        raise ProcessingError("采样率应为 8 kHz 至 192 kHz 的整数。")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or len(audio) < 64:
        raise ProcessingError("需要至少 64 个采样点的单声道或立体声音频。")
    if not np.isfinite(audio).all():
        raise ProcessingError("音轨包含无效采样值，无法拆分。")
    if float(np.max(np.abs(audio))) < 1e-9:
        raise ProcessingError("所选音轨接近静音，没有足够特征可供拆分。")
    return audio


def _thread_context():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=2, user_api="blas")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)


def _nmf(v: np.ndarray, rank: int, cancelled: Cancelled | None,
         progress: Progress) -> tuple[np.ndarray, np.ndarray]:
    """Bounded-training-frame KL NMF followed by fixed-dictionary activation fit."""
    rng = np.random.default_rng(314159)
    energy = np.sum(v, axis=0)
    active = np.flatnonzero(energy > max(float(np.max(energy)) * .002, 1e-9))
    if len(active) < 2:
        raise ProcessingError("有效声音太短，无法学习多个声学特征；请选择更长的片段。")
    if len(active) > 900:
        active = active[np.linspace(0, len(active) - 1, 900).astype(int)]
    train = np.ascontiguousarray(v[:, active])
    rank = min(rank, len(active), v.shape[0])
    # Random positive initialization avoids assigning a preselected instrument name
    # or importing a dictionary from the user's demonstration recording.
    w = rng.uniform(.1, 1., (v.shape[0], rank)).astype(np.float32)
    w /= np.maximum(np.sum(w, axis=0, keepdims=True), 1e-10)
    h = rng.uniform(.1, 1., (rank, len(active))).astype(np.float32)
    h *= np.mean(train) * train.shape[0] / rank
    for iteration in range(85):
        _check_cancel(cancelled)
        ratio = train / np.maximum(w @ h, 1e-8)
        h *= (w.T @ ratio) / np.maximum(np.sum(w, axis=0)[:, None], 1e-8)
        ratio = train / np.maximum(w @ h, 1e-8)
        w *= (ratio @ h.T) / np.maximum(np.sum(h, axis=1)[None, :], 1e-8)
        scale = np.maximum(np.sum(w, axis=0), 1e-8)
        w /= scale[None, :]
        h *= scale[:, None]
        if iteration % 10 == 0:
            progress(10 + 42 * iteration / 85, f"学习本段录音的频谱字典（{iteration + 1}/85）…")
    # Fit the learned dictionary to every frame. No temporal subsampling of the
    # final masks: events between training frames are still processed.
    activation = np.maximum(w.T @ v, 1e-7).astype(np.float32)
    denominator = np.maximum(np.sum(w, axis=0)[:, None], 1e-8)
    for iteration in range(55):
        _check_cancel(cancelled)
        activation *= (w.T @ (v / np.maximum(w @ activation, 1e-8))) / denominator
        if iteration % 10 == 0:
            progress(52 + 20 * iteration / 55, f"追踪各频谱特征的时间变化（{iteration + 1}/55）…")
    return w, activation


def _group_atoms(w: np.ndarray, h: np.ndarray, count: int) -> np.ndarray:
    """Weighted grouping uses both coactivation and coarse log-frequency shape."""
    rank = w.shape[1]
    count = min(count, rank)
    # Smooth and resample envelopes before comparing: co-activating harmonics
    # belong together more often than neighboring frequencies do.
    envelope = ndimage.uniform_filter1d(h, size=3, axis=1, mode="nearest")
    positions = np.linspace(0, h.shape[1] - 1, min(256, h.shape[1])).astype(int)
    temporal = _normalize_rows(np.sqrt(np.maximum(envelope[:, positions], 0)))
    edges = np.unique(np.rint(np.geomspace(1, w.shape[0], 33)).astype(int))
    edges = np.r_[0, edges]
    coarse = np.stack([np.sum(w[a:b], axis=0) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    spectral = _normalize_rows(np.sqrt(np.maximum(coarse, 0)))
    # Flatness and activation duty distinguish broadband short attacks from
    # sustained tonal energy while retaining the spectrum/envelope evidence.
    centroid = np.sum(w * np.linspace(0, 1, len(w))[:, None], axis=0) / np.maximum(w.sum(0), 1e-8)
    entropy = -np.sum(w * np.log(np.maximum(w, 1e-9)), axis=0) / np.log(len(w))
    duty = np.mean(h > np.max(h, axis=1, keepdims=True) * .18, axis=1)
    stats = np.column_stack((centroid, entropy, duty))
    stats = (stats - stats.mean(0)) / np.maximum(stats.std(0), .1)
    features = np.concatenate((temporal * np.sqrt(.60), spectral * np.sqrt(.30), stats * .13), axis=1)
    weights = np.maximum(np.sum(h, axis=1), 1e-8)
    weights /= np.max(weights)
    centers = [features[int(np.argmax(weights))]]
    # Deterministic weighted farthest-point initialization; no trained semantic
    # classifier and no fixed bands masquerading as learned source tracks.
    for _ in range(1, count):
        distance = np.min(np.sum((features[:, None] - np.asarray(centers)[None]) ** 2, axis=2), axis=1)
        centers.append(features[int(np.argmax(distance * np.sqrt(weights)))])
    centers = np.asarray(centers).copy()
    labels = np.zeros(rank, dtype=int)
    for _ in range(35):
        distances = np.sum((features[:, None] - centers[None]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        # A very simple recording can yield identical atoms. Still retain explicit
        # components without an empty group or a divide-by-zero during masking.
        for cluster in range(count):
            if not np.any(new_labels == cluster):
                sizes = np.bincount(new_labels, minlength=count)
                candidates = np.flatnonzero(sizes[new_labels] > 1)
                worst = candidates[np.argmax(distances[candidates, new_labels[candidates]] * weights[candidates])]
                new_labels[worst] = cluster
        if np.array_equal(labels, new_labels) and _ > 0:
            break
        labels = new_labels
        for cluster in range(count):
            selected = labels == cluster
            centers[cluster] = np.average(features[selected], axis=0, weights=weights[selected])
    return labels


def split_features(audio: np.ndarray, sr: int, components: int = 4,
                   progress: Progress | None = None,
                   cancelled: Cancelled | None = None) -> list[tuple[str, np.ndarray]]:
    """Return 2–8 auditionable NMF groups with exact mixture reconstruction.

    The shared real-valued masks retain input stereo phase relationships. The
    final floating-point reconstruction residual is allocated to the last group,
    so enabling every group recreates its parent instead of silently changing it.
    """
    _check_cancel(cancelled)
    audio = _validate(audio, sr)
    if not isinstance(components, (int, np.integer)) or isinstance(components, bool) or not 2 <= components <= 8:
        raise ProcessingError("特征拆分轨数应为 2 至 8 的整数。")
    say = progress or (lambda _p, _m: None)
    nfft = int(2 ** round(np.log2(sr * .046)))
    nfft = max(128, min(8192, nfft))
    while nfft > len(audio):
        nfft //= 2
    hop = nfft // 4
    frames = int(np.ceil(len(audio) / hop)) + 1
    # Guard before allocating complex spectra or multiplying multi-track arrays.
    estimated_bytes = (audio.nbytes * (components + 3)
                       + (nfft // 2 + 1) * frames * (audio.shape[1] * 8 + 32 + 4 * components))
    if estimated_bytes > 650 * 1024 ** 2:
        raise ProcessingError("本次特征拆分的内存需求过大，请将音频分成较短片段或减少轨数。")
    say(3, "读取双声道频谱，学习可供人工辨识的声音特征…")
    _, _, spectrum = signal.stft(audio.T, fs=sr, nperseg=nfft, noverlap=nfft-hop,
                                 boundary="zeros", padded=True, axis=-1)
    magnitude = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=0))
    # Partial whitening keeps quieter high-frequency events learnable without
    # amplifying bins whose only content is numerical noise.
    mean = np.mean(magnitude, axis=1)
    scale = np.maximum(mean, float(np.max(mean)) * .002 + 1e-9) ** .6
    v = np.ascontiguousarray(magnitude / scale[:, None], dtype=np.float32)
    maximum = max(float(np.max(v)), 1e-9)
    v /= maximum
    with _thread_context():
        w, h = _nmf(v, max(16, components * 6), cancelled, say)
        if w.shape[1] < components:
            raise ProcessingError("这段音频可用的时间特征不足，请减少拆分轨数或选择更长片段。")
        # Whitening is useful for learning quiet events, but grouping atoms by
        # whitened energy overweights tiny broadband artifacts. Restore the
        # recording's frequency scale for clustering and its energy weights.
        grouping_w = w * scale[:, None]
        grouping_norm = np.maximum(grouping_w.sum(axis=0), 1e-9)
        labels = _group_atoms(grouping_w / grouping_norm[None, :],
                              h * grouping_norm[:, None], components)
        _check_cancel(cancelled)
        # A moderate exponent reduces cross-talk while keeping masks continuous.
        denominator = np.zeros_like(v)
        estimates = []
        for cluster in range(components):
            estimate = np.maximum(w[:, labels == cluster] @ h[labels == cluster], 0) ** 1.4
            estimates.append(estimate)
            denominator += estimate
        denominator = np.maximum(denominator, 1e-15)
        tracks = []
        running = np.zeros_like(audio)
        for cluster, estimate in enumerate(estimates):
            _check_cancel(cancelled)
            say(76 + 23 * cluster / components, f"重建特征候选轨（{cluster + 1}/{components}）…")
            if cluster == components - 1:
                track = audio - running
            else:
                masked = spectrum * (estimate / denominator)[None]
                _, recovered = signal.istft(masked, fs=sr, nperseg=nfft, noverlap=nfft-hop,
                                             boundary=True, time_axis=-1, freq_axis=-2)
                track = np.asarray(recovered.T[:len(audio)], dtype=np.float32)
                if len(track) < len(audio):
                    track = np.pad(track, ((0, len(audio) - len(track)), (0, 0)))
                running += track
            tracks.append((f"特征候选 {cluster + 1}", np.asarray(track, dtype=np.float32)))
    say(100, "已拆分为特征候选轨，请逐轨独听后命名和选择。")
    return tracks


def split_repeating(audio: np.ndarray, sr: int, progress: Progress | None = None,
                    cancelled: Cancelled | None = None) -> list[tuple[str, np.ndarray]]:
    """Extract repeated transient templates as a candidate stem plus remainder.

    Uses the existing multi-timbre template fitter with a zero neural guide.
    This path is explicitly signal processing, not a new neural inference. The
    reference recording itself supplies the repeated timbre, timing and phase.
    """
    _check_cancel(cancelled)
    audio = _validate(audio, sr)
    if len(audio) < sr:
        raise ProcessingError("重复节拍拆分至少需要 1 秒，并含至少 5 个相似事件。")
    from clickremover.multiclass import remove_guided
    say = progress or (lambda _p, _m: None)

    def report(value: float, message: str) -> None:
        _check_cancel(cancelled)
        # Legacy routine labels include neural protection, which is deliberately
        # absent here. Report the actual operation performed by this adapter.
        text = "分析重复事件及不同节拍音色…" if value < 84 else "从重复事件学习模板并拟合候选节拍轨…"
        say(max(5., min(90., (value - 70) * 5)), text)

    say(2, "寻找可重复的瞬态声音；它可能是节拍器，也可能是乐器起音…")
    cleaned, details = remove_guided(audio, sr, ProcessConfig(strength=1.0, method="classic"),
                                    np.zeros_like(audio), progress=report)
    _check_cancel(cancelled)
    candidate = audio - cleaned
    if details["processed_count"] < 5 or float(np.max(np.abs(candidate))) < 1e-8:
        raise ProcessingError("未提取到足够可靠的重复声音；请尝试特征拆分或更长的片段。")
    say(100, f"已提取 {details['processed_count']} 次重复事件，请独听确认其是否为节拍器。")
    return [("重复节拍候选", candidate.astype(np.float32)),
            ("重复节拍之外", cleaned.astype(np.float32))]
