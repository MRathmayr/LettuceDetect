"""JSON reporter for benchmark results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.benchmarks.core.models import BenchmarkResults


class JSONReporter:
    """Generate JSON reports from benchmark results."""

    def __init__(self, output_dir: str = "tests/benchmarks/results"):
        """Initialize reporter.

        Args:
            output_dir: Directory to write reports to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        results: list[BenchmarkResults],
        filename: str | None = None,
        include_predictions: bool = False,
    ) -> Path:
        """Save benchmark results to JSON file.

        Args:
            results: List of BenchmarkResults to save
            filename: Output filename (default: benchmark_{timestamp}.json)
            include_predictions: Include individual predictions (large file)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        output = {
            "timestamp": datetime.now().isoformat(),
            "n_benchmarks": len(results),
            "results": [],
        }

        for r in results:
            result_dict = r.to_dict()
            if include_predictions:
                result_dict["predictions"] = [
                    {
                        "sample_id": p.sample_id,
                        "ground_truth": p.ground_truth,
                        "predicted_score": p.predicted_score,
                        "predicted_label": p.predicted_label,
                        "latency_ms": p.latency_ms,
                    }
                    for p in r.predictions
                ]
            output["results"].append(result_dict)

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return output_path

    def save_summary(
        self,
        results: list[BenchmarkResults],
        filename: str = "benchmark_summary.json",
    ) -> Path:
        """Save a summary of benchmark results.

        Args:
            results: List of BenchmarkResults
            filename: Output filename

        Returns:
            Path to saved file
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        for r in results:
            key = f"{r.component}_{r.dataset}"
            summary["components"][key] = {
                "auroc": r.metrics.auroc,
                "f1": r.metrics.f1,
                "optimal_f1": r.metrics.optimal_f1,
                "latency_mean_ms": r.timing.mean_ms,
                "latency_p95_ms": r.timing.p95_ms,
                "n_samples": r.metrics.n_samples,
            }

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        return output_path

    def load_results(self, filepath: str | Path) -> dict[str, Any]:
        """Load benchmark results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary with benchmark data
        """
        with open(filepath) as f:
            return json.load(f)


def main():
    """CLI entrypoint for generating reports from existing results."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file with raw results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/benchmarks/results",
        help="Output directory",
    )
    args = parser.parse_args()

    reporter = JSONReporter(args.output_dir)

    if args.input:
        data = reporter.load_results(args.input)
        print(f"Loaded {len(data.get('results', []))} results")


if __name__ == "__main__":
    main()
