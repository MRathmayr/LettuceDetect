"""Memory tracking utilities for benchmarking."""

from contextlib import contextmanager

from tests.benchmarks.core.models import MemoryStats

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryTracker:
    """Track GPU and RAM memory usage during benchmark runs."""

    def __init__(self, track_gpu: bool = True, track_ram: bool = True):
        """Initialize memory tracker.

        Args:
            track_gpu: Whether to track GPU memory (requires torch with CUDA)
            track_ram: Whether to track RAM usage (requires psutil)
        """
        self.track_gpu = track_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.track_ram = track_ram and PSUTIL_AVAILABLE

        self._gpu_start: float = 0.0
        self._gpu_peak: float = 0.0
        self._ram_start: float = 0.0
        self._ram_end: float = 0.0

    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory allocated in MB."""
        if not self.track_gpu:
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)

    def _get_gpu_peak_mb(self) -> float:
        """Get peak GPU memory allocated in MB."""
        if not self.track_gpu:
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def _get_ram_mb(self) -> float:
        """Get current process RAM usage in MB."""
        if not self.track_ram:
            return 0.0
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def reset_peak(self) -> None:
        """Reset GPU peak memory counter."""
        if self.track_gpu:
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def track(self):
        """Context manager for tracking memory during a code block.

        Usage:
            tracker = MemoryTracker()
            with tracker.track():
                # code to measure
            stats = tracker.get_stats()
        """
        # Reset peak counter for accurate measurement
        if self.track_gpu:
            torch.cuda.reset_peak_memory_stats()
            self._gpu_start = self._get_gpu_memory_mb()

        if self.track_ram:
            self._ram_start = self._get_ram_mb()

        try:
            yield
        finally:
            if self.track_gpu:
                self._gpu_peak = self._get_gpu_peak_mb()

            if self.track_ram:
                self._ram_end = self._get_ram_mb()

    def get_stats(self) -> MemoryStats:
        """Get memory statistics from last tracked block."""
        gpu_peak = self._gpu_peak if self.track_gpu else None
        gpu_allocated = self._get_gpu_memory_mb() if self.track_gpu else None
        ram_delta = (self._ram_end - self._ram_start) if self.track_ram else None

        return MemoryStats(
            gpu_peak_mb=gpu_peak,
            gpu_allocated_mb=gpu_allocated,
            ram_delta_mb=ram_delta,
        )


def get_gpu_memory_mb() -> float | None:
    """Get current GPU memory allocated in MB, or None if unavailable."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_gpu_peak_memory_mb() -> float | None:
    """Get peak GPU memory allocated in MB, or None if unavailable."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def clear_gpu_cache() -> None:
    """Clear GPU cache to get consistent memory measurements."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
