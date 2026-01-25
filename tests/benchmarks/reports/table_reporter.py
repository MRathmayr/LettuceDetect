"""Table reporter for human-readable benchmark results."""

from pathlib import Path

from tests.benchmarks.core.models import BenchmarkResults


class TableReporter:
    """Generate human-readable table reports from benchmark results."""

    def __init__(self, output_dir: str = "tests/benchmarks/results"):
        """Initialize reporter.

        Args:
            output_dir: Directory to write reports to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_results(self, results: list[BenchmarkResults]) -> str:
        """Format benchmark results as a table.

        Args:
            results: List of BenchmarkResults

        Returns:
            Formatted table string
        """
        lines = []
        lines.append("=" * 100)
        lines.append("BENCHMARK RESULTS")
        lines.append("=" * 100)
        lines.append("")

        # Header
        header = f"{'Component':<20} {'Dataset':<15} {'AUROC':>8} {'F1':>8} {'Opt F1':>8} {'Latency':>10} {'P95':>10} {'Samples':>8}"
        lines.append(header)
        lines.append("-" * 100)

        # Results rows
        for r in results:
            auroc = f"{r.metrics.auroc:.3f}" if r.metrics.auroc else "N/A"
            f1 = f"{r.metrics.f1:.3f}" if r.metrics.f1 else "N/A"
            opt_f1 = f"{r.metrics.optimal_f1:.3f}" if r.metrics.optimal_f1 else "N/A"
            latency = f"{r.timing.mean_ms:.1f}ms"
            p95 = f"{r.timing.p95_ms:.1f}ms"
            samples = str(r.metrics.n_samples)

            row = f"{r.component:<20} {r.dataset:<15} {auroc:>8} {f1:>8} {opt_f1:>8} {latency:>10} {p95:>10} {samples:>8}"
            lines.append(row)

        lines.append("-" * 100)
        lines.append("")

        return "\n".join(lines)

    def format_detailed(self, result: BenchmarkResults) -> str:
        """Format a single benchmark result with full details.

        Args:
            result: BenchmarkResults to format

        Returns:
            Detailed formatted string
        """
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"{result.component} on {result.dataset}")
        lines.append(f"{'='*60}")
        lines.append("")

        # Accuracy metrics
        lines.append("Accuracy Metrics:")
        lines.append(f"  AUROC:           {result.metrics.auroc:.4f}" if result.metrics.auroc else "  AUROC:           N/A")
        if result.metrics.auroc_ci_lower and result.metrics.auroc_ci_upper:
            lines.append(f"  AUROC 95% CI:    [{result.metrics.auroc_ci_lower:.4f}, {result.metrics.auroc_ci_upper:.4f}]")
        lines.append(f"  Accuracy:        {result.metrics.accuracy:.4f}" if result.metrics.accuracy else "  Accuracy:        N/A")
        lines.append(f"  Precision:       {result.metrics.precision:.4f}" if result.metrics.precision else "  Precision:       N/A")
        lines.append(f"  Recall:          {result.metrics.recall:.4f}" if result.metrics.recall else "  Recall:          N/A")
        lines.append(f"  F1:              {result.metrics.f1:.4f}" if result.metrics.f1 else "  F1:              N/A")
        lines.append("")

        lines.append("Optimal Threshold Metrics:")
        lines.append(f"  Threshold:       {result.metrics.optimal_threshold:.4f}" if result.metrics.optimal_threshold else "  Threshold:       N/A")
        lines.append(f"  Optimal F1:      {result.metrics.optimal_f1:.4f}" if result.metrics.optimal_f1 else "  Optimal F1:      N/A")
        lines.append(f"  MCC:             {result.metrics.mcc:.4f}" if result.metrics.mcc else "  MCC:             N/A")
        lines.append(f"  Balanced Acc:    {result.metrics.balanced_accuracy:.4f}" if result.metrics.balanced_accuracy else "  Balanced Acc:    N/A")
        lines.append(f"  Specificity:     {result.metrics.specificity:.4f}" if result.metrics.specificity else "  Specificity:     N/A")
        lines.append("")

        # Timing metrics
        lines.append("Timing Metrics:")
        lines.append(f"  Mean:            {result.timing.mean_ms:.2f}ms")
        lines.append(f"  Std:             {result.timing.std_ms:.2f}ms")
        lines.append(f"  P50:             {result.timing.p50_ms:.2f}ms")
        lines.append(f"  P95:             {result.timing.p95_ms:.2f}ms")
        lines.append(f"  Min:             {result.timing.min_ms:.2f}ms")
        lines.append(f"  Max:             {result.timing.max_ms:.2f}ms")
        if result.timing.cold_start_ms:
            lines.append(f"  Cold Start:      {result.timing.cold_start_ms:.2f}ms")
        lines.append("")

        # Memory metrics
        if result.memory:
            lines.append("Memory Metrics:")
            if result.memory.gpu_peak_mb:
                lines.append(f"  GPU Peak:        {result.memory.gpu_peak_mb:.1f}MB")
            if result.memory.gpu_allocated_mb:
                lines.append(f"  GPU Allocated:   {result.memory.gpu_allocated_mb:.1f}MB")
            if result.memory.ram_delta_mb:
                lines.append(f"  RAM Delta:       {result.memory.ram_delta_mb:.1f}MB")
            lines.append("")

        # Sample info
        lines.append("Sample Information:")
        lines.append(f"  Total Samples:   {result.metrics.n_samples}")
        lines.append(f"  Hallucinations:  {result.metrics.n_hallucinations}")
        lines.append(f"  Factual:         {result.metrics.n_factual}")
        lines.append("")

        # Config
        if result.config:
            lines.append("Configuration:")
            for k, v in result.config.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    def save_table(
        self,
        results: list[BenchmarkResults],
        filename: str = "benchmark_table.txt",
    ) -> Path:
        """Save formatted table to file.

        Args:
            results: List of BenchmarkResults
            filename: Output filename

        Returns:
            Path to saved file
        """
        table = self.format_results(results)
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(table)
        return output_path

    def print_results(self, results: list[BenchmarkResults]) -> None:
        """Print formatted table to stdout.

        Args:
            results: List of BenchmarkResults
        """
        print(self.format_results(results))

    def print_detailed(self, result: BenchmarkResults) -> None:
        """Print detailed result to stdout.

        Args:
            result: BenchmarkResults to print
        """
        print(self.format_detailed(result))


def main():
    """CLI entrypoint for generating table reports."""
    import argparse
    import json

    from tests.benchmarks.core.models import AccuracyMetrics, TimingStats

    parser = argparse.ArgumentParser(description="Generate table reports")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/benchmarks/results",
        help="Output directory",
    )
    args = parser.parse_args()

    # Load results
    with open(args.input) as f:
        data = json.load(f)

    # Convert to BenchmarkResults
    results = []
    for r in data.get("results", []):
        metrics_data = r.get("metrics", {})
        timing_data = r.get("timing", {})

        auroc_ci = metrics_data.get("auroc_ci_95", [None, None])

        metrics = AccuracyMetrics(
            auroc=metrics_data.get("auroc"),
            auroc_ci_lower=auroc_ci[0] if auroc_ci else None,
            auroc_ci_upper=auroc_ci[1] if auroc_ci else None,
            accuracy=metrics_data.get("accuracy"),
            precision=metrics_data.get("precision"),
            recall=metrics_data.get("recall"),
            f1=metrics_data.get("f1"),
            optimal_threshold=metrics_data.get("optimal_threshold"),
            optimal_f1=metrics_data.get("optimal_f1"),
            mcc=metrics_data.get("mcc"),
            balanced_accuracy=metrics_data.get("balanced_accuracy"),
            specificity=metrics_data.get("specificity"),
            brier_score=metrics_data.get("brier_score"),
            n_samples=metrics_data.get("n_samples", 0),
            n_hallucinations=metrics_data.get("n_hallucinations", 0),
            n_factual=metrics_data.get("n_factual", 0),
        )

        timing = TimingStats(
            mean_ms=timing_data.get("mean_ms", 0),
            std_ms=timing_data.get("std_ms", 0),
            min_ms=timing_data.get("min_ms", 0),
            max_ms=timing_data.get("max_ms", 0),
            p50_ms=timing_data.get("p50_ms", 0),
            p90_ms=timing_data.get("p90_ms", 0),
            p95_ms=timing_data.get("p95_ms", 0),
            cold_start_ms=timing_data.get("cold_start_ms"),
            n_samples=timing_data.get("n_samples", 0),
        )

        results.append(
            BenchmarkResults(
                component=r.get("component", "unknown"),
                dataset=r.get("dataset", "unknown"),
                predictions=[],
                metrics=metrics,
                timing=timing,
                config=r.get("config", {}),
            )
        )

    reporter = TableReporter(args.output_dir)
    reporter.print_results(results)


if __name__ == "__main__":
    main()
