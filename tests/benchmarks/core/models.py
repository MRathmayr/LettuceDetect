"""Data models for benchmark results."""

from dataclasses import dataclass, field


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    sample_id: str
    ground_truth: int  # 0=factual, 1=hallucination
    predicted_score: float  # 0.0-1.0 hallucination probability
    predicted_label: int  # 0 or 1 at threshold
    latency_ms: float
    component_name: str


@dataclass
class TimingStats:
    """Timing statistics from benchmark runs."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    cold_start_ms: float | None = None
    n_samples: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "cold_start_ms": self.cold_start_ms,
            "n_samples": self.n_samples,
        }


@dataclass
class AccuracyMetrics:
    """Accuracy metrics from benchmark evaluation."""

    auroc: float | None
    auroc_ci_lower: float | None
    auroc_ci_upper: float | None
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    optimal_threshold: float | None
    optimal_f1: float | None
    mcc: float | None
    balanced_accuracy: float | None
    specificity: float | None
    brier_score: float | None
    n_samples: int
    n_hallucinations: int
    n_factual: int
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "auroc": self.auroc,
            "auroc_ci_95": [self.auroc_ci_lower, self.auroc_ci_upper]
            if self.auroc_ci_lower is not None
            else None,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "optimal_threshold": self.optimal_threshold,
            "optimal_f1": self.optimal_f1,
            "mcc": self.mcc,
            "balanced_accuracy": self.balanced_accuracy,
            "specificity": self.specificity,
            "brier_score": self.brier_score,
            "n_samples": self.n_samples,
            "n_hallucinations": self.n_hallucinations,
            "n_factual": self.n_factual,
            "error": self.error,
        }


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    gpu_peak_mb: float | None = None
    gpu_allocated_mb: float | None = None
    ram_delta_mb: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "gpu_peak_mb": self.gpu_peak_mb,
            "gpu_allocated_mb": self.gpu_allocated_mb,
            "ram_delta_mb": self.ram_delta_mb,
        }


@dataclass
class BenchmarkResults:
    """Complete results from a benchmark run."""

    component: str
    dataset: str
    predictions: list[PredictionResult]
    metrics: AccuracyMetrics
    timing: TimingStats
    memory: MemoryStats | None = None
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "dataset": self.dataset,
            "metrics": self.metrics.to_dict(),
            "timing": self.timing.to_dict(),
            "memory": self.memory.to_dict() if self.memory else None,
            "config": self.config,
            "n_predictions": len(self.predictions),
        }
