#!/usr/bin/env python
"""Repository-root entry point.

The pipeline itself lives at ``seismos_rag/run_pipeline.py``; this shim exists so both
``python run_pipeline.py`` (from the project root) and
``python -m seismos_rag.run_pipeline`` work identically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # seismos_rag
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # core seismos package

from seismos_rag.run_pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
