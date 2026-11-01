"""Private local subprocess used to keep PyTorch's OpenMP runtime isolated."""

import sys

import numpy as np

from .neural import estimate_piano


def main() -> int:
    if len(sys.argv) != 4:
        print("需要输入、采样率和输出文件。", flush=True)
        return 2
    try:
        audio = np.load(sys.argv[1], allow_pickle=False)
        rate = int(sys.argv[2])
        estimate = estimate_piano(
            audio, rate,
            progress=lambda percent, message: print(f"PROGRESS\t{percent}\t{message}", flush=True),
        )
        np.save(sys.argv[3], estimate, allow_pickle=False)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
