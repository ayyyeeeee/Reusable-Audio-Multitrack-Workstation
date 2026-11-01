"""神经钢琴估计引导的多音色重复事件分离。

本模块本身是自适应信号处理，不负责加载或运行神经网络。调用方必须
传入与原音对齐的神经钢琴估计；该估计同时影响事件分类及模板保护。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .engine import (
    ProcessConfig,
    ProcessingError,
    _analysis_signal,
    _auto_query,
    _find_matches,
    _learn_template,
    _local_match,
    _reference_query,
    _subtract,
    _tempo_consistency,
    _validate_config,
    _window_fade,
)


def _correct_timing(y: np.ndarray, sr: int, query: np.ndarray, pre: int,
                    centers: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Recover a displaced match only when adjacent intervals agree and a strong match exists."""
    corrected = centers.copy()
    new_scores = scores.copy()
    changes = 0
    for i in range(2, len(centers) - 2):
        left, right = centers[i - 1], centers[i + 1]
        period = (right - left) / 2
        if period <= 0:
            continue
        expected = (left + right) / 2
        displacement = abs(centers[i] - expected)
        if not (max(.025 * sr, .04 * period) < displacement < .15 * period):
            continue
        tolerance = max(.020 * sr, .06 * period)
        before = centers[i - 1] - centers[i - 2]
        after = centers[i + 2] - centers[i + 1]
        if abs(before - period) > tolerance or abs(after - period) > tolerance:
            continue
        score, start = _local_match(y, query, int(round(expected)), pre, round(.012 * sr))
        if score >= .70 and score >= .85 * scores[i]:
            corrected[i] = start + pre
            new_scores[i] = score
            changes += 1
    return corrected, new_scores, changes


def _classify(audio: np.ndarray, piano: np.ndarray, sr: int, centers: np.ndarray,
              channel: int) -> tuple[np.ndarray, dict | None]:
    """根据局部频谱强弱与强组相位一致性做一次保守的二类划分。

    若缺少足够事件或无明显分组，则所有事件沿用同一模板。这里不是
    通用的乐器识别器：钢琴也可能产生重复且相位稳定的瞬态。
    """
    labels = np.zeros(len(centers), dtype=np.int32)
    if len(centers) < 8:
        return labels, None
    length = round(.1 * sr)
    valid = (centers >= 0) & (centers + length <= len(audio))
    selected = np.flatnonzero(valid)
    if len(selected) < 8:
        return labels, None
    # A long recording may have many beats: cap the training subset, then
    # classify every event using the chosen frequency and threshold.
    training = selected
    if len(training) > 400:
        training = training[np.linspace(0, len(training) - 1, 400).astype(int)]
    window = np.hanning(length).astype(np.float32)
    freq = np.fft.rfftfreq(length, 1 / sr)

    def features(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windows = np.stack([
            (audio[centers[i]:centers[i] + length, channel]
             - .4 * piano[centers[i]:centers[i] + length, channel]) * window
            for i in indices
        ])
        spectra = np.fft.rfft(windows, axis=1)
        high = (freq >= min(3000., sr * .30)) & (freq <= min(6000., sr * .46))
        if not np.any(high):
            high = freq > sr * .25
        scale = np.sqrt(np.mean(np.abs(spectra[:, high]) ** 2, axis=1)) + 1e-12
        return spectra, scale

    spectra, scales = features(training)
    normalized = np.abs(spectra) / scales[:, None]
    candidates = np.flatnonzero((freq >= 750) & (freq <= min(6500., sr * .46)))
    best: tuple[float, int, float, float, float, int] | None = None
    for k in candidates:
        order = np.argsort(normalized[:, k])
        values = normalized[order, k]
        # Each proposed class needs at least four training events.
        for cut in range(4, len(values) - 3):
            low, high = values[:cut], values[cut:]
            ratio = float(np.median(high) / (np.median(low) + 1e-12))
            gap = float(high[0] / (low[-1] + 1e-12))
            if ratio < 4.0 or gap < 1.8:
                continue
            strong = spectra[order[cut:], k]
            phase = float(abs(np.mean(strong / np.maximum(abs(strong), 1e-12))))
            if phase < .8:
                continue
            power = float(np.median(np.abs(strong)))
            score = np.log(ratio) * np.log(gap) * phase * np.sqrt(power)
            if best is None or score > best[0]:
                threshold = float(np.sqrt(max(low[-1], 1e-12) * high[0]))
                best = (float(score), int(k), threshold, ratio, phase, len(high))
    if best is None:
        return labels, None
    _, k, threshold, ratio, phase, _ = best
    training_high = normalized[:, k] > threshold
    # A very quiet boundary fragment can obtain a high normalized ratio even
    # though its absolute tone is absent. Do not assign such a fragment to a
    # loud accent template solely because its broadband denominator is small.
    absolute_floor = .25 * float(np.median(np.abs(spectra[training_high, k])))
    # Process bounded batches to avoid materializing every event of a long file.
    for start in range(0, len(selected), 128):
        batch = selected[start:start + 128]
        values, batch_scales = features(batch)
        labels[batch] = ((np.abs(values[:, k]) / batch_scales > threshold)
                         & (np.abs(values[:, k]) >= absolute_floor)).astype(np.int32)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts < 4):
        return np.zeros(len(centers), dtype=np.int32), None
    return labels, {"frequency_hz": float(freq[k]), "ratio": ratio,
                    "phase_coherence": phase, "counts": counts.tolist()}


def _protect_template(template: np.ndarray, before: int, centers: np.ndarray,
                      fit_events: list[dict], piano: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """与神经钢琴预测一致的频率最多保护 15%；不可用时原样返回模板。"""
    length = len(template)
    gains = np.asarray([event["relative_gain"] for event in fit_events], dtype=float)
    positive = gains[gains > 0]
    if len(positive) < 4:
        return template, 0.0
    middle = float(np.median(positive))
    usable = []
    for center, gain in zip(centers, gains):
        start = int(center) - before
        if start >= 0 and start + length <= len(piano) and gain > middle * .2:
            usable.append(piano[start:start + length] / gain)
    if len(usable) < 4:
        return template, 0.0
    if len(usable) > 64:
        usable = [usable[i] for i in np.linspace(0, len(usable) - 1, 64).astype(int)]
    piano_median = np.median(np.stack(usable), axis=0)
    if np.linalg.norm(piano_median) < 1e-8:
        return template, 0.0
    fade = _window_fade(length, sr)
    protected = np.empty_like(template)
    scores = []
    for ch in range(template.shape[1]):
        t_spec = np.fft.rfft(template[:, ch])
        p_spec = np.fft.rfft(piano_median[:, ch] * fade)
        t_abs, p_abs = np.abs(t_spec), np.abs(p_spec)
        phase_agreement = np.clip(np.real(p_spec * np.conj(t_spec)) / (p_abs * t_abs + 1e-12), 0, 1)
        relative_power = np.clip(p_abs / (t_abs + 1e-12), 0, 1)
        agreement = ndimage.gaussian_filter1d(phase_agreement * relative_power,
                                               max(1., 25 / (sr / length)))
        agreement = np.clip(agreement, 0, 1)
        # Spectral weighting bounds the correction's L2 energy to 15% of the
        # template; taper only the correction so a zero estimate is an exact no-op.
        correction = np.fft.irfft(t_spec * (-.15 * agreement), n=length)
        protected[:, ch] = (template[:, ch] + correction * fade).astype(np.float32)
        scores.append(float(np.sum(agreement * t_abs) / (np.sum(t_abs) + 1e-12)))
    return protected, float(np.mean(scores))


def _refine_tone(audio: np.ndarray, result: np.ndarray, piano: np.ndarray, sr: int,
                 centers: np.ndarray, template: np.ndarray, before: int,
                 frequency: float, strength: float) -> np.ndarray:
    """Fit a repeatable tonal accent's amplitude and phase independently per beat.

    The two quadrature bases come from the learned (neural-protected) class
    template; corrections are local and bounded by the source projection.
    """
    if strength <= 0 or len(centers) < 4:
        return result
    half_width = max(40., min(120., frequency * .035))
    lo, hi = frequency - half_width, frequency + half_width
    if lo <= 0 or hi >= sr * .48:
        return result
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    basis_a = signal.sosfiltfilt(sos, template, axis=0)
    basis_b = np.imag(signal.hilbert(basis_a, axis=0))
    fade = _window_fade(len(template), sr)[:, None]
    basis_a *= fade
    basis_b *= fade
    basis = np.stack((basis_a.ravel(), basis_b.ravel()), axis=1)
    gram = basis.T @ basis
    if not np.isfinite(gram).all() or np.linalg.cond(gram) > 1e6 or np.trace(gram) < 1e-12:
        return result
    gram = gram + np.eye(2) * 1e-12
    target = signal.sosfiltfilt(sos, result, axis=0)
    original = signal.sosfiltfilt(sos, audio, axis=0)
    piano_band = signal.sosfiltfilt(sos, piano, axis=0)
    output = result.copy()
    for center in centers:
        start = int(center) - before
        end = start + len(template)
        if start < 0 or end > len(audio):
            continue
        residual_coef = np.linalg.solve(gram, basis.T @ target[start:end].reshape(-1))
        original_coef = np.linalg.solve(gram, basis.T @ original[start:end].reshape(-1))
        piano_coef = np.linalg.solve(gram, basis.T @ piano_band[start:end].reshape(-1))
        # Preserve a bounded, same-phase part of the neural piano estimate also
        # during refinement; otherwise the second fit could undo that protection.
        if np.dot(piano_coef, original_coef) > 0:
            piano_protection = .15 * piano_coef
            psize, pcap = np.linalg.norm(piano_protection), .15 * np.linalg.norm(original_coef)
            if psize > pcap and psize > 0:
                piano_protection *= pcap / psize
            residual_coef -= piano_protection
        # Never correct more than the source's projection onto this template.
        cap, size = np.linalg.norm(original_coef), np.linalg.norm(residual_coef)
        if size > cap and size > 0:
            residual_coef *= cap / size
        output[start:end] -= (.85 * strength *
                              (residual_coef[0] * basis_a + residual_coef[1] * basis_b))
    return output


def remove_guided(audio: np.ndarray, sr: int, config: ProcessConfig,
                  piano_estimate: np.ndarray, reference: tuple[np.ndarray, int] | None = None,
                  progress=None) -> tuple[np.ndarray, dict]:
    """用已对齐的神经钢琴估计引导分类及模板保护，保守减除重复声。"""
    _validate_config(config)
    audio = np.asarray(audio, dtype=np.float32)
    piano = np.asarray(piano_estimate, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if piano.ndim == 1:
        piano = piano[:, None]
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or len(audio) < sr:
        raise ProcessingError("需要至少 1 秒的单声道或立体声音频。")
    if sr < 8000 or sr > 192000 or not np.isfinite(audio).all():
        raise ProcessingError("音频采样率或数据无效。")
    if piano.shape != audio.shape or not np.isfinite(piano).all():
        raise ProcessingError("神经钢琴估计须与输入长度、声道完全一致，且不含无效值。")
    say = progress or (lambda _p, _m: None)
    say(73, "分析录音中的瞬态与重复模式…")
    y, analysis_sr, channel = _analysis_signal(audio, sr)
    bpm = None
    if reference is None:
        query, pre, bpm = _auto_query(y, analysis_sr, config)
    else:
        ref, ref_sr = reference
        query, pre = _reference_query(np.asarray(ref, dtype=np.float32), ref_sr, analysis_sr)
    say(78, "识别相似节拍事件并估计不同音色…")
    matches, scores = _find_matches(y, analysis_sr, query, pre)
    if len(matches) < 5:
        raise ProcessingError("匹配到的节拍声不足 5 次；未生成输出。请检查参考音与原录音的设备和环境是否一致。")
    if reference is None:
        consistency, _ = _tempo_consistency(matches / analysis_sr, config.min_bpm, config.max_bpm)
        if consistency < .50:
            raise ProcessingError("检测结果的节奏一致性不足，未生成输出；可提供节拍器参考音。")
    matches, scores, timing_changes = _correct_timing(y, analysis_sr, query, pre, matches, scores)
    centers = np.rint(matches.astype(np.float64) * sr / analysis_sr).astype(int)
    global_template, global_before, centers, good = _learn_template(audio, sr, centers, channel)
    centers, scores = centers[good], scores[good]
    labels, class_info = _classify(audio, piano, sr, centers, channel)
    warnings = ["神经钢琴估计可能包含节拍声或漏掉钢琴；请对照原音试听，尤其检查钢琴起音。"]
    if timing_changes:
        warnings.append(f"根据邻近节拍和波形相关性微调了 {timing_changes} 个事件位置，建议试听复核。")
    if class_info is None:
        warnings.append("未发现足够可靠的不同节拍音色，采用单一事件模板。")
    else:
        warnings.append("检测到不同的重复音色并分别处理；其他音色仍可能漏检。")
    say(84, "按音色学习模板，并使用神经钢琴估计保护相应频率…")
    removed = np.zeros_like(audio)
    all_events: list[dict] = []
    protections = []
    accent_refinement = None
    for label in np.unique(labels):
        selected = np.flatnonzero(labels == label)
        class_centers, class_scores = centers[selected], scores[selected]
        if len(np.unique(labels)) == 1:
            template, before = global_template, global_before
        else:
            try:
                template, before, class_centers, good_class = _learn_template(audio, sr, class_centers, channel)
                class_centers, class_scores = class_centers[good_class], class_scores[good_class]
            except ProcessingError:
                template, before = global_template, global_before
                warnings.append("一个音色类别的完整匹配不足，已退回共同模板。")
        # Fit against the unprotected template so refitting cannot cancel the
        # model-guided piano protection applied to the subtraction template.
        _, fits = _subtract(audio, sr, class_centers, class_scores, template, before, 1.0)
        protected, protection = _protect_template(template, before, class_centers, fits, piano, sr)
        protections.append(protection)
        class_output, events = _subtract(audio, sr, class_centers, class_scores,
                                         protected, before, config.strength, fit_template=template)
        removed += audio - class_output
        if class_info is not None and label == 1:
            accent_refinement = (class_centers, protected, before)
        for event in events:
            event["sound_class"] = int(label)
        all_events.extend(events)
    all_events.sort(key=lambda event: event["time_seconds"])
    if not protections or max(protections) < .005:
        warnings.append("神经钢琴估计与模板的一致性不足，钢琴频率保护可能作用很小。")
    count = sum(event["processed"] for event in all_events)
    if count == 0:
        warnings.append("没有减除任何事件；如果力度设为 0，输出将与输入相同。")
    cleaned = audio - removed
    if accent_refinement is not None:
        class_centers, template, before = accent_refinement
        # Overlapping event windows make the independently fitted correction ambiguous.
        if len(centers) > 1 and np.min(np.diff(np.sort(centers))) < len(template):
            warnings.append("节拍事件过于密集，跳过高音重拍的逐拍相位修正。")
        else:
            cleaned = _refine_tone(audio, cleaned, piano, sr, class_centers, template, before,
                                   class_info["frequency_hz"], config.strength)
    return cleaned, {"processed_count": count, "candidate_count": len(all_events),
                             "estimated_bpm": bpm, "events": all_events, "warnings": warnings,
                             "class_info": class_info, "protection_strength": protections}
