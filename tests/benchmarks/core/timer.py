"""High-precision timing utilities for benchmarking."""

import time
from contextlib import contextmanager
from typing import Callable

import numpy as np

from tests.benchmarks.core.models import TimingStats

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BenchmarkTimer:
    """High-precision timer with GPU synchronization support."""

    def __init__(self, sync_cuda: bool = True):
        """Initialize timer.

        Args:
            sync_cuda: If True, synchronize CUDA before timing for accurate GPU measurements.
        """
        self.sync_cuda = sync_cuda and TORCH_AVAILABLE and torch.cuda.is_available()
        self._timings: list[float] = []
        self._cold_start: float | None = None

    def _sync(self) -> None:
        """Synchronize CUDA if enabled."""
        if self.sync_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def measure(self):
        """Context manager for timing a code block.

        Usage:
            timer = BenchmarkTimer()
            with timer.measure():
                # code to time
            print(timer.last_ms)
        """
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._timings.append(elapsed_ms)

    def time_function(
        self,
        func: Callable,
        *args,
        warmup_runs: int = 3,
        timed_runs: int = 10,
        **kwargs,
    ) -> TimingStats:
        """Time a function with warmup.

        Args:
            func: Function to time
            *args: Positional arguments for func
            warmup_runs: Number of warmup runs (discarded)
            timed_runs: Number of timed runs
            **kwargs: Keyword arguments for func

        Returns:
            TimingStats with timing information
        """
        # Warmup runs
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # Timed runs
        self._timings = []
        for _ in range(timed_runs):
            with self.measure():
                func(*args, **kwargs)

        return self.get_stats()

    def record(self, elapsed_ms: float) -> None:
        """Record an external timing measurement."""
        self._timings.append(elapsed_ms)

    def record_cold_start(self, elapsed_ms: float) -> None:
        """Record cold start timing (first invocation)."""
        self._cold_start = elapsed_ms

    @property
    def last_ms(self) -> float:
        """Get the last recorded timing in milliseconds."""
        if not self._timings:
            return 0.0
        return self._timings[-1]

    def get_stats(self) -> TimingStats:
        """Compute timing statistics from recorded timings."""
        if not self._timings:
            return TimingStats(
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                cold_start_ms=self._cold_start,
                n_samples=0,
            )

        arr = np.array(self._timings)
        return TimingStats(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            cold_start_ms=self._cold_start,
            n_samples=len(self._timings),
        )

    def reset(self) -> None:
        """Reset all recorded timings."""
        self._timings = []
        self._cold_start = None


def sync_cuda_if_available() -> None:
    """Synchronize CUDA if available."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()
