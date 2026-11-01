"""Single local worker and reversible separation publication."""
from pathlib import Path
import json
import threading
import traceback
import numpy as np
from clickremover.engine import ProcessingError
from .project import ProjectStore, atomic_json

LABELS = {"vocals": "人声 · 候选", "drums": "鼓与打击 · 候选", "bass": "贝斯 · 候选",
          "guitar": "吉他 · 候选", "piano": "钢琴 · 候选", "other": "其他声音 · 候选"}


def consistent_stems(audio, estimates):
    if len(estimates) < 2:
        raise ProcessingError("神经模型没有返回多轨结果。")
    total = np.zeros_like(audio, dtype=np.float64)
    stems = []
    for key, estimate in estimates.items():
        if estimate.shape != audio.shape or not np.isfinite(estimate).all():
            raise ProcessingError("神经模型输出不对齐或包含无效采样。")
        estimate = np.asarray(estimate, dtype=np.float32)
        total += estimate
        stems.append((LABELS.get(key, key + " · 候选"), estimate))
    stems.append(("校正残差 · 需试听", (audio.astype(np.float64) - total).astype(np.float32)))
    return stems


class StudioService:
    def __init__(self, root: Path):
        self.store = ProjectStore(root)
        self.lock = threading.RLock()
        self.active = None
        self.busy = False
        self.progress = 0.
        self.message = "导入一段录音，开始分轨。"
        self.error = None
        self.cancel_event = threading.Event()
        self.worker = None
        last = self.store.root / "last-project.json"
        if last.is_file():
            try:
                identifier = json.loads(last.read_text(encoding="utf-8"))["id"]
                self.store.load(identifier)
                self.active = identifier
                self.message = "已打开上次保存的工程，可继续试听、分轨和导出。"
            except (ValueError, KeyError, ProcessingError):
                pass

    def remember(self, identifier):
        self.active = identifier
        atomic_json(self.store.root / "last-project.json", {"id": identifier})

    def state(self):
        with self.lock:
            session = None
            if self.active:
                try:
                    session = self.store.public(self.store.load(self.active))
                except ProcessingError as exc:
                    self.error = str(exc)
                    self.message = "当前工程无法读取；可导入音频或打开其他保存的工程。"
                    self.active = None
            return {"application": "stemstudio", "workspace": str(self.store.root),
                    "busy": self.busy, "progress": self.progress, "message": self.message,
                    "error": self.error,
                    "session": session,
                    "sessions": self.store.list_projects()}

    def require_idle(self):
        if self.busy:
            raise ProcessingError("正在处理音频，请完成或取消后再操作。")

    def require_active(self):
        if not self.active:
            raise ProcessingError("请先导入音频或打开工程。")
        return self.active

    def import_file(self, path, name=None):
        with self.lock:
            self.require_idle()
            self.busy, self.error, self.message = True, None, "正在解码音频…"
        try:
            data = self.store.import_file(path, name)
            with self.lock:
                self.remember(data["id"])
                self.progress, self.message = 0., "音频已导入，选择分离方式。"
            return self.store.public(data)
        except Exception as exc:
            with self.lock:
                self.error, self.message = str(exc), "导入未完成，之前的工程保留。"
            raise
        finally:
            with self.lock:
                self.busy = False

    def open(self, identifier):
        with self.lock:
            self.require_idle()
            data = self.store.load(identifier)
            self.remember(identifier)
            self.error, self.progress, self.message = None, 0., "已打开保存的工程。"
            return self.store.public(data)

    def update(self, values, name=None):
        with self.lock:
            self.require_idle()
            return self.store.public(self.store.update(self.require_active(), values, name))

    def undo(self):
        with self.lock:
            self.require_idle()
            data = self.store.undo(self.require_active())
            self.message, self.error = "已撤销上次分离，恢复当时的音轨与混音设置。", None
            return self.store.public(data)

    def separate(self, method, components=6, track_id=None):
        if method not in ("demucs", "practice", "features", "repeating") or (method == "repeating" and not track_id):
            raise ProcessingError("请选择支持的分离方式。")
        if not isinstance(components, int) or isinstance(components, bool) or not 2 <= components <= 8:
            raise ProcessingError("特征分轨数量应为 2–8。")
        with self.lock:
            self.require_idle()
            identifier = self.require_active()
            data = self.store.load(identifier)
            if track_id and not any(t["id"] == track_id for t in data["tracks"]):
                raise ProcessingError("待拆分音轨不存在。")
            self.busy, self.progress, self.error = True, 0., None
            self.message = "准备分离；CPU 模型推理可能需要数分钟。"
            self.cancel_event.clear()
            self.worker = threading.Thread(target=self._run, args=(identifier, method, components, track_id),
                                           name="stemstudio-worker", daemon=True)
            self.worker.start()

    def _run(self, identifier, method, components, track_id):
        def say(value, message):
            with self.lock:
                self.progress = max(0., min(98., float(value) * .98))
                self.message = message
        try:
            data = self.store.load(identifier)
            track = next((t for t in data["tracks"] if t["id"] == track_id), None)
            audio = self.store.track_audio(data, track) if track else self.store.original(data)
            sr = data["sample_rate"]
            if method in ("demucs", "practice"):
                from .demucs_backend import separate_demucs
                neural_input = audio
                prefix = []
                if method == "practice":
                    from .features import split_repeating
                    try:
                        repeating = split_repeating(audio, sr, progress=lambda p, m: say(p * .2, m),
                                                    cancelled=self.cancel_event.is_set)
                    except ProcessingError as exc:
                        raise ProcessingError(f"练习录音模式无法提取可靠重复节拍：{exc} 可改选神经网络六类音轨。") from exc
                    prefix = [("节拍器 / 重复起音 · 候选", repeating[0][1])]
                    neural_input = repeating[1][1]
                neural_progress = (lambda p, m: say(20 + p * .8, m)) if prefix else say
                estimates = separate_demucs(neural_input, sr, progress=neural_progress,
                                            cancelled=self.cancel_event.is_set)
                stems = prefix + consistent_stems(neural_input, estimates)
                notes = ["Demucs 六类候选与校正残差。标签需试听确认，婴儿声可能分散在人声和其他轨中。",
                         "全部轨开启、音量100%时恢复原音；静音选定轨道后才会去掉对应声音。",
                         "钢琴、节拍器可能串音，可选中混合轨继续按特征或重复节拍拆分。"]
                if prefix:
                    notes.insert(0, "先从整段录音提取重复起音，再对剩余声音做神经网络分轨；第一轨请独听确认后再静音。")
            else:
                from .features import split_features, split_repeating
                if method == "features":
                    stems = split_features(audio, sr, components=components, progress=say,
                                           cancelled=self.cancel_event.is_set)
                else:
                    stems = split_repeating(audio, sr, progress=say, cancelled=self.cancel_event.is_set)
                notes = ["候选轨按录音特征学习分组，名称不代表已识别的具体乐器或人声。",
                         "请逐轨独听和命名；同一声音可能在多条轨中，必要时继续拆分。"]
            if self.cancel_event.is_set():
                raise ProcessingError("已取消，原工程音轨保留。")
            say(99, "保存音轨并检查全轨重建…")
            with self.lock:
                if self.cancel_event.is_set():
                    raise ProcessingError("已取消，原工程音轨保留。")
                self.store.commit_stems(identifier, stems, method, parent_id=track_id, notes=notes)
                self.progress, self.message = 100., f"完成：生成 {len(stems)} 条可试听音轨。请独听确认后静音。"
        except Exception as exc:
            with self.lock:
                if self.cancel_event.is_set():
                    self.error = None
                    self.message = "已取消，之前的音轨与混音设置保留。"
                else:
                    self.error = str(exc)
                    self.message = "分离未完成；之前的音轨与混音设置保留。"
                (self.store.root / "last-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            with self.lock:
                self.busy = False

    def cancel(self):
        with self.lock:
            if self.busy:
                self.cancel_event.set()
                self.message = "正在取消；已保存音轨保留。"

    def export(self, values, format, mode):
        with self.lock:
            self.require_idle()
            identifier = self.require_active()
            self.busy, self.message, self.error = True, "正在导出…", None
        try:
            result = self.store.export(identifier, values, format, mode)
            with self.lock:
                self.message = result["message"]
            return result
        except Exception as exc:
            with self.lock:
                self.error, self.message = str(exc), "导出未完成，工程音轨与混音设置保留。"
            raise
        finally:
            with self.lock:
                self.busy = False
