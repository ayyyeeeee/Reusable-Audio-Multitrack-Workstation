"""Loopback-only HTTP interface. Audio never leaves the local computer."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit, quote
import webbrowser
import urllib.request
from clickremover.engine import ProcessingError
from .service import StudioService

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service):
        self.service = service
        super().__init__(address, StudioHandler)


class StudioHandler(BaseHTTPRequestHandler):
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def log_message(self, format, *args):
        if str(args[1] if len(args) > 1 else "").startswith("5"):
            super().log_message(format, *args)

    def trusted_request(self):
        port = self.server.server_port
        expected = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in expected:
            self.json_response({"error": "仅允许本机工作台访问。"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in expected}:
            self.json_response({"error": "不允许跨站请求。"}, 403)
            return False
        return True

    def json_response(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def body(self, limit):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ProcessingError("无效请求长度。") from None
        if not 0 < size <= limit:
            raise ProcessingError("文件为空或超过限制（音频上传最多256 MiB）。")
        content = self.rfile.read(size)
        if len(content) != size:
            raise ProcessingError("上传不完整。")
        return content

    def do_POST(self):
        if not self.trusted_request():
            return
        path = urlsplit(self.path).path
        service = self.server.service
        try:
            if path == "/api/import":
                filename = unquote(self.headers.get("X-Filename", ""))
                if not filename:
                    raise ProcessingError("缺少文件名。")
                suffix = Path(filename).suffix.lower()
                if suffix not in (".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".aiff", ".aif", ".mp4", ".opus", ".wma"):
                    raise ProcessingError("请选择常见音频文件（WAV / MP3 / M4A / FLAC 等）。")
                content = self.body(256 * 1024 * 1024)
                with tempfile.TemporaryDirectory(prefix="stemstudio-import-") as temporary:
                    source = Path(temporary) / ("input" + suffix)
                    source.write_bytes(content)
                    data = service.import_file(source, filename)
                self.json_response(data)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ProcessingError("接口需要 JSON 请求。")
            data = json.loads(self.body(1024 * 1024))
            if not isinstance(data, dict):
                raise ProcessingError("无效请求参数。")
            if path == "/api/separate":
                service.separate(data.get("method", "demucs"), data.get("components", 6))
                self.json_response({"accepted": True}, 202)
            elif path == "/api/split":
                if not isinstance(data.get("track_id"), str):
                    raise ProcessingError("请选择待拆分音轨。")
                service.separate(data.get("method", "features"), data.get("components", 4), data["track_id"])
                self.json_response({"accepted": True}, 202)
            elif path == "/api/session":
                self.json_response(service.update(data.get("tracks"), data.get("name")))
            elif path == "/api/open":
                self.json_response(service.open(data.get("session_id")))
            elif path == "/api/undo":
                self.json_response(service.undo())
            elif path == "/api/cancel":
                service.cancel()
                self.json_response({"cancelled": True})
            elif path == "/api/export":
                self.json_response(service.export(data.get("tracks"), data.get("format", "wav"), data.get("mode", "mix")))
            else:
                self.json_response({"error": "接口不存在。"}, 404)
        except (ProcessingError, ValueError, TypeError) as exc:
            self.json_response({"error": str(exc)}, 400)
        except Exception as exc:
            self.json_response({"error": f"处理失败：{exc}"}, 500)

    def do_GET(self):
        if not self.trusted_request():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/state":
                self.json_response(self.server.service.state())
                return
            if path in ("/", "/index.html", "/app.js", "/style.css"):
                self.send_file(WEB / ("index.html" if path == "/" else path[1:]))
                return
            match = re.fullmatch(r"/audio/([0-9a-f]{32})/(original\.wav|original\.peaks\.json|[0-9a-f]{32}\.(?:wav|peaks\.json))", path)
            if match:
                self.send_file(self.server.service.store.directory(match[1]) / match[2])
                return
            match = re.fullmatch(r"/download/([0-9a-f]{32}\.(?:wav|flac|mp3|zip))", path)
            if match:
                self.send_file(self.server.service.store.root / "exports" / match[1], download=True)
                return
            self.json_response({"error": "文件不存在。"}, 404)
        except (OSError, ProcessingError) as exc:
            self.json_response({"error": str(exc)}, 404)

    def send_file(self, path, download=False):
        if not path.is_file():
            self.json_response({"error": "文件不存在。"}, 404)
            return
        length = path.stat().st_size
        start, stop, partial = 0, length - 1, False
        request_range = self.headers.get("Range")
        if request_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request_range)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            if match[1]:
                start = int(match[1])
                stop = min(int(match[2]) if match[2] else stop, stop)
            else:
                start = max(0, length - int(match[2]))
            if start > stop or start >= length:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{length}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            partial = True
        self.send_response(206 if partial else 200)
        mime = {".js": "text/javascript", ".wav": "audio/wav"}.get(path.suffix,
                       mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(stop - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store" if path.suffix in (".html", ".js", ".css") else "private, max-age=86400")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{stop}/{length}")
        if download:
            name = path.name
            metadata = path.with_suffix(".download.json")
            if metadata.is_file():
                name = json.loads(metadata.read_text(encoding="utf-8")).get("filename", name)
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(name))
        self.end_headers()
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = stop - start + 1
                while remaining:
                    block = stream.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main():
    parser = argparse.ArgumentParser(description="本地多轨声音分离工作台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--import-file", type=Path)
    parser.add_argument("--projects", type=Path, default=ROOT / "音轨工程")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    # A second double-click opens the existing workspace instead of failing to bind.
    if args.port and not args.import_file:
        try:
            with urllib.request.urlopen(url + "/api/state", timeout=1) as response:
                existing = json.load(response)
            if (existing.get("application") == "stemstudio"
                    and Path(existing["workspace"]).resolve() == args.projects.resolve()):
                print(f"工作台已经运行：{url}", flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return
        except (OSError, ValueError, KeyError):
            pass
    service = StudioService(args.projects)
    if args.import_file:
        service.import_file(args.import_file)
    server = StudioHTTPServer(("127.0.0.1", args.port), service)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"声音分轨工作台：{url}\n工程保存在：{service.store.root}\n关闭此窗口或按 Ctrl+C 停止服务。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        service.cancel()
        if service.worker:
            service.worker.join(timeout=10)
        server.server_close()


if __name__ == "__main__":
    main()
