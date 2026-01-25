"""Pydantic configuration models for benchmarking."""

from typing import Literal

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Configuration for a benchmark dataset."""

    name: Literal["ragtruth", "halueval_qa", "all"] = "ragtruth"
    limit: int | None = Field(
        default=None, description="Maximum samples to load (None = all)"
    )
    cache_dir: str | None = Field(
        default=None, description="Directory for caching loaded datasets"
    )


class ComponentBenchmarkConfig(BaseModel):
    """Configuration for benchmarking a single component."""

    warmup_runs: int = Field(default=3, ge=0, description="Warmup runs to discard")
    timed_runs: int = Field(default=10, ge=1, description="Runs to measure timing")
    batch_size: int = Field(default=1, ge=1, description="Batch size for inference")
    compute_ci: bool = Field(
        default=True, description="Compute bootstrap CI (slower but more informative)"
    )
    ci_bootstraps: int = Field(default=1000, ge=100, description="Bootstrap samples for CI")
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Classification threshold"
    )


class BenchmarkConfig(BaseModel):
    """Main benchmark configuration."""

    # Mode
    quick: bool = Field(
        default=False,
        description="Quick mode: 100 samples, no CI computation",
    )
    full: bool = Field(
        default=False,
        description="Full mode: all samples, full metrics",
    )

    # Dataset
    datasets: list[DatasetConfig] = Field(
        default_factory=lambda: [DatasetConfig()],
        description="Datasets to benchmark",
    )

    # Component settings
    component: ComponentBenchmarkConfig = Field(
        default_factory=ComponentBenchmarkConfig,
        description="Component benchmark settings",
    )

    # Output
    output_dir: str = Field(
        default="tests/benchmarks/results",
        description="Directory for benchmark results",
    )
    save_predictions: bool = Field(
        default=False,
        description="Save individual predictions (large files)",
    )

    # Hardware
    device: str = Field(default="cuda", description="Device for inference")
    sync_cuda: bool = Field(
        default=True, description="Sync CUDA for accurate timing"
    )
    track_memory: bool = Field(
        default=True, description="Track GPU/RAM memory usage"
    )

    def apply_quick_mode(self) -> "BenchmarkConfig":
        """Apply quick mode settings."""
        if not self.quick:
            return self
        for ds in self.datasets:
            if ds.limit is None or ds.limit > 100:
                ds.limit = 100
        self.component.compute_ci = False
        self.component.warmup_runs = 1
        self.component.timed_runs = 3
        return self

    @classmethod
    def for_quick_test(cls, dataset: str = "ragtruth") -> "BenchmarkConfig":
        """Create config for quick testing."""
        return cls(
            quick=True,
            datasets=[DatasetConfig(name=dataset, limit=100)],
            component=ComponentBenchmarkConfig(
                warmup_runs=1,
                timed_runs=3,
                compute_ci=False,
            ),
        )

    @classmethod
    def for_full_benchmark(cls) -> "BenchmarkConfig":
        """Create config for full benchmark."""
        return cls(
            full=True,
            datasets=[DatasetConfig(name="all")],
            component=ComponentBenchmarkConfig(
                warmup_runs=3,
                timed_runs=10,
                compute_ci=True,
            ),
        )
