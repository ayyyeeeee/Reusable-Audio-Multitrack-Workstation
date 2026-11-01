"""Entry point for the reusable local mixing studio."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
deps = root / ".deps"
if deps.is_dir():
    sys.path.insert(1, str(deps))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if __name__ == "__main__":
    from stemstudio.server import main
    main()
