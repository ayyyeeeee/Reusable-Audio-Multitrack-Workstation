"""Build a separately saved demonstration project through the public service."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stemstudio.service import StudioService

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
args = parser.parse_args()
service = StudioService(ROOT / "音轨工程")
data = service.import_file(args.input)
start = time.monotonic()
service.separate("practice")
service.worker.join()
if service.error:
    raise RuntimeError(service.error)
data = service.store.load(data["id"])
data = service.store.update(data["id"], data["tracks"], "示例 · 节拍与乐器分轨")
report = {"session_id": data["id"], "seconds": time.monotonic() - start,
          "tracks": [{"name": t["name"], "rms": t["rms"]} for t in data["tracks"]]}
report_dir = ROOT / ".analysis"
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / "practice-demo-report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False), flush=True)
