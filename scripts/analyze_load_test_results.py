#!/usr/bin/env python3
"""
Analyze and visualize load test results.

This script reads JSON reports from load tests and generates:
- Summary statistics
- Performance trends
- Comparison reports
- Visual charts (if matplotlib available)
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


class LoadTestAnalyzer:
    """Analyzer for load test results."""

    def __init__(self, results_dir: Path):
        """Initialize analyzer.

        Args:
            results_dir: Directory containing test result JSON files
        """
        self.results_dir = results_dir
        self.console = Console()

    def print(self, message: str, style: Optional[str] = None):
        """Print message with optional styling."""
        self.console.print(message, style=style)

    def load_results(self) -> List[Dict]:
        """Load all test result files.

        Returns:
            List of test result dictionaries
        """
        results = []
        json_files = sorted(self.results_dir.glob("load_test_report_*.json"))

        if not json_files:
            self.print("No test results found", "yellow")
            return results

        for json_file in json_files:
            try:
                with json_file.open() as f:
                    data = json.load(f)
                    # Only include successful test runs
                    if data.get("summary", {}).get("total", 0) > 0:
                        data["filename"] = json_file.name
                        data["timestamp"] = datetime.fromtimestamp(data["created"])
                        results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                self.print(f"Error loading {json_file.name}: {e}", "red")

        return results

    def get_latest_result(self, results: List[Dict]) -> Optional[Dict]:
        """Get the most recent test result.

        Args:
            results: List of test results

        Returns:
            Latest test result or None
        """
        if not results:
            return None
        return max(results, key=lambda x: x["created"])

    def print_summary(self, result: Dict):
        """Print summary of a single test result.

        Args:
            result: Test result dictionary
        """
        summary = result.get("summary", {})
        timestamp = result.get("timestamp", datetime.now())

        self.console.print(Panel.fit(
            f"[bold]Test Results - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/bold]",
            border_style="blue",
        ))

        # Summary table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        table.add_row("Total Tests:", f"[blue]{summary.get('total', 0)}[/blue]")
        table.add_row("Passed:", f"[green]{summary.get('passed', 0)}[/green]")
        table.add_row("Failed:", f"[red]{summary.get('failed', 0)}[/red]")
        table.add_row("Duration:", f"[blue]{result.get('duration', 0):.2f}s[/blue]")

        self.console.print(table)

    def print_test_details(self, result: Dict):
        """Print detailed test results.

        Args:
            result: Test result dictionary
        """
        tests = result.get("tests", [])
        if not tests:
            return

        print("\n")
        table = Table(title="Test Details")
        table.add_column("Test", style="cyan", no_wrap=False)
        table.add_column("Status", style="green")
        table.add_column("Duration", justify="right")

        for test in tests:
            nodeid = test.get("nodeid", "").split("::")[-1]
            outcome = test.get("outcome", "unknown")
            duration = test.get("call", {}).get("duration", 0)

            status_style = "green" if outcome == "passed" else "red"
            table.add_row(
                nodeid,
                f"[{status_style}]{outcome}[/{status_style}]",
                f"{duration:.2f}s"
            )

        self.console.print(table)

    def compare_results(self, results: List[Dict]):
        """Compare multiple test results.

        Args:
            results: List of test results
        """
        if len(results) < 2:
            self.print("Need at least 2 test results for comparison", "yellow")
            return

        print("\n")
        self.console.print(Panel.fit(
            "[bold]Performance Trends[/bold]",
            border_style="blue",
        ))

        # Extract metrics over time
        timestamps = [r["timestamp"] for r in results]
        total_tests = [r.get("summary", {}).get("total", 0) for r in results]
        passed_tests = [r.get("summary", {}).get("passed", 0) for r in results]
        durations = [r.get("duration", 0) for r in results]

        # Calculate statistics
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        success_rates = [
            (p / t * 100) if t > 0 else 0
            for p, t in zip(passed_tests, total_tests)
        ]
        avg_success_rate = statistics.mean(success_rates) if success_rates else 0

        # Print statistics
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        table.add_row("Test Runs:", f"[blue]{len(results)}[/blue]")
        table.add_row("Avg Duration:", f"[blue]{avg_duration:.2f}s[/blue]")
        table.add_row("Min Duration:", f"[green]{min_duration:.2f}s[/green]")
        table.add_row("Max Duration:", f"[yellow]{max_duration:.2f}s[/yellow]")
        table.add_row("Avg Success Rate:", f"[green]{avg_success_rate:.1f}%[/green]")

        self.console.print(table)

        # Recent trend
        if len(results) >= 3:
            recent_durations = durations[-3:]
            trend = "improving" if recent_durations[-1] < recent_durations[0] else "degrading"
            trend_pct = abs((recent_durations[-1] - recent_durations[0]) / recent_durations[0] * 100)

            print()
            if trend == "improving":
                self.print(f"✓ Performance improving: {trend_pct:.1f}% faster", "green")
            else:
                self.print(f"⚠ Performance degrading: {trend_pct:.1f}% slower", "yellow")

    def print_test_breakdown(self, result: Dict):
        """Print breakdown by test category.

        Args:
            result: Test result dictionary
        """
        tests = result.get("tests", [])
        if not tests:
            return

        # Group by test class
        categories = {}
        for test in tests:
            nodeid = test.get("nodeid", "")
            if "::" in nodeid:
                parts = nodeid.split("::")
                if len(parts) >= 2:
                    category = parts[1]  # Test class name
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(test)

        print("\n")
        table = Table(title="Test Categories")
        table.add_column("Category", style="cyan")
        table.add_column("Tests", justify="right")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Avg Duration", justify="right")

        for category, cat_tests in sorted(categories.items()):
            total = len(cat_tests)
            passed = sum(1 for t in cat_tests if t.get("outcome") == "passed")
            failed = total - passed
            avg_duration = statistics.mean(
                [t.get("call", {}).get("duration", 0) for t in cat_tests]
            )

            table.add_row(
                category,
                str(total),
                str(passed),
                str(failed),
                f"{avg_duration:.2f}s"
            )

        self.console.print(table)

    def print_performance_sparkline(self, results: List[Dict]) -> None:
        """Print ASCII sparkline of performance trends.

        Args:
            results: List of test results
        """
        if len(results) < 2:
            return

        durations = [r.get("duration", 0) for r in results]
        
        # Normalize to 0-8 range for sparkline characters
        min_val = min(durations)
        max_val = max(durations)
        range_val = max_val - min_val if max_val > min_val else 1
        
        # Sparkline characters (8 levels)
        chars = "▁▂▃▄▅▆▇█"
        
        sparkline = ""
        for duration in durations:
            normalized = (duration - min_val) / range_val
            index = min(int(normalized * 7), 7)
            sparkline += chars[index]
        
        print("\n")
        self.print("Performance Trend (Duration):", "cyan")
        self.print(f"  {sparkline}", "blue")
        self.print(f"  Min: {min_val:.2f}s  Max: {max_val:.2f}s", "dim")

    def export_csv(self, results: List[Dict], output_file: Path):
        """Export results to CSV.

        Args:
            results: List of test results
            output_file: Path to save CSV
        """
        import csv

        with Path(output_file).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Total Tests',
                'Passed',
                'Failed',
                'Duration (s)',
                'Success Rate (%)'
            ])

            for result in results:
                timestamp = result["timestamp"].strftime('%Y-%m-%d %H:%M:%S')
                summary = result.get("summary", {})
                total = summary.get("total", 0)
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                duration = result.get("duration", 0)
                success_rate = (passed / total * 100) if total > 0 else 0

                writer.writerow([
                    timestamp,
                    total,
                    passed,
                    failed,
                    f"{duration:.2f}",
                    f"{success_rate:.1f}"
                ])

        self.print(f"✓ CSV exported to: {output_file}", "green")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze load test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d", "--dir",
        type=Path,
        default=Path("tests/results/load_tests"),
        help="Directory containing test results (default: tests/results/load_tests)",
    )
    parser.add_argument(
        "-l", "--latest",
        action="store_true",
        help="Show only the latest test result",
    )
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="Compare all test results",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        help="Export results to CSV",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed test information",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = LoadTestAnalyzer(args.dir)

    # Load results
    analyzer.print(f"Loading results from: {args.dir}", "blue")
    results = analyzer.load_results()

    if not results:
        analyzer.print("No test results found", "red")
        return 1

    analyzer.print(f"Found {len(results)} test result(s)\n", "green")

    # Show latest result
    if args.latest or not args.compare:
        latest = analyzer.get_latest_result(results)
        if latest:
            analyzer.print_summary(latest)
            if args.verbose:
                analyzer.print_test_details(latest)
                analyzer.print_test_breakdown(latest)

    # Compare results
    if args.compare and len(results) > 1:
        analyzer.compare_results(results)
        analyzer.print_performance_sparkline(results)

    # Export CSV
    if args.csv:
        analyzer.export_csv(results, args.csv)

    return 0


if __name__ == "__main__":
    exit(main())
