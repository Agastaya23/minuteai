"""MinuteAI — grounded meeting intelligence."""

import os
import sys

if sys.platform == "darwin":
    # torch (sentence-transformers) and faiss each bundle an OpenMP runtime; on macOS,
    # loading both in one process segfaults under load. One OpenMP thread avoids it.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
