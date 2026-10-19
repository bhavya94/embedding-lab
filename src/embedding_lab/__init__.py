"""Embedding retrieval experiments on NFCorpus."""

import os

# FAISS and torch each bring their own OpenMP runtime on macOS. With both loaded and several threads, the
# process deadlocks. One thread avoids it; it also makes FAISS latency a single-core number.
os.environ.setdefault("OMP_NUM_THREADS", "1")
# The tokenizers library's Rust thread pool can deadlock MPS graph compilation during training; one thread is
# enough for these short texts.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
