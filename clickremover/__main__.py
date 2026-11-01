"""命令行入口：python -m clickremover 输入.m4a -o 输出.wav。"""

import argparse
import json
import sys

from .engine import ProcessConfig, process_file


def main() -> int:
    parser = argparse.ArgumentParser(description="使用本地神经网络和自适应模板减除钢琴录音中的节拍器声。")
    parser.add_argument("input", help="输入音频；支持 FFmpeg 能读取的单声道或立体声文件")
    parser.add_argument("-o", "--output", required=True, help="新的输出 WAV 文件")
    parser.add_argument("--strength", type=float, default=.95, help="处理力度 0–1，默认 0.95")
    parser.add_argument("--method", choices=("neural", "classic"), default="neural",
                        help="神经网络或传统模板；默认 neural（神经模型限非商业用途）")
    parser.add_argument("--min-bpm", type=float, default=50, help="最小节拍速度，默认 50")
    parser.add_argument("--max-bpm", type=float, default=200, help="最大节拍速度，默认 200")
    parser.add_argument("--reference", help="仅包含节拍器声的短音频，可选")
    parser.add_argument("--no-timestamps", action="store_true", help="不生成事件 CSV")
    args = parser.parse_args()
    config = ProcessConfig(strength=args.strength, min_bpm=args.min_bpm, max_bpm=args.max_bpm,
                           reference_path=args.reference, export_timestamps=not args.no_timestamps,
                           method=args.method)
    try:
        report = process_file(args.input, args.output, config=config,
                              progress=lambda percent, message: print(f"{percent:5.1f}% {message}", file=sys.stderr))
    except Exception as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in report.items() if key != "events"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
