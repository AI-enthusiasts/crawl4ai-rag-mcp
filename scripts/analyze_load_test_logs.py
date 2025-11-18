#!/usr/bin/env python3
"""
Analyze load test logs and create a summary report of all issues.

This script:
1. Runs load tests
2. Captures all output and errors
3. Analyzes logs for problems
4. Creates a comprehensive report file
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class LoadTestAnalyzer:
    """Analyzes load test results and logs."""

    def __init__(self, output_dir: str = "tests/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = self.output_dir / f"load_test_report_{self.timestamp}.md"

        self.issues = {
            "errors": [],
            "warnings": [],
            "timeouts": [],
            "failures": [],
            "performance_issues": [],
            "connection_issues": [],
        }

        self.metrics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "total_duration": 0.0,
        }

    def run_load_tests(self, test_path: str | None = None, quick: bool = True) -> tuple[str, str, int]:
        """Run load tests and capture output.

        Args:
            test_path: Specific test to run (None = all tests)
            quick: If True, skip slow tests

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        cmd = ["uv", "run", "pytest", "-v", "-s"]

        if test_path:
            cmd.append(test_path)
            # Don't add markers when specific test is selected
        else:
            cmd.append("tests/integration/test_mcp_load_testing.py")
            # Add markers only when running all tests
            if quick:
                cmd.extend(["-m", "load and not slow"])
            else:
                cmd.extend(["-m", "load"])

        # Add JSON report
        json_report = self.output_dir / f"test_results_{self.timestamp}.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])

        print(f"Running: {' '.join(cmd)}")
        print(f"Output will be saved to: {self.report_file}")

        try:
            result = subprocess.run(
                cmd,
                check=False, capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                cwd=Path.cwd(),
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as e:
            return (
                e.stdout.decode() if e.stdout else "",
                e.stderr.decode() if e.stderr else "",
                -1,
            )

    def analyze_output(self, stdout: str, stderr: str):
        """Analyze test output for issues."""
        # Combine output
        full_output = stdout + "\n" + stderr

        # Extract test results
        test_pattern = r"(PASSED|FAILED|ERROR|SKIPPED)\s+\[(\d+)%\]"
        for match in re.finditer(test_pattern, full_output):
            status, _ = match.groups()
            self.metrics["total_tests"] += 1
            if status == "PASSED":
                self.metrics["passed_tests"] += 1
            elif status in ["FAILED", "ERROR"]:
                self.metrics["failed_tests"] += 1

        # Find errors
        error_patterns = [
            (r"ERROR.*?(?=\n\n|\Z)", "errors"),
            (r"FAILED.*?(?=\n\n|\Z)", "failures"),
            (r"TimeoutError.*?(?=\n\n|\Z)", "timeouts"),
            (r"WARNING.*?(?=\n\n|\Z)", "warnings"),
            (r"ConnectionError.*?(?=\n\n|\Z)", "connection_issues"),
            (r"RuntimeError.*?(?=\n\n|\Z)", "errors"),
        ]

        for pattern, category in error_patterns:
            matches = re.findall(pattern, full_output, re.DOTALL)
            for match in matches:
                if match not in self.issues[category]:
                    self.issues[category].append(match.strip())

        # Find performance issues
        perf_patterns = [
            r"success_rate.*?(\d+\.?\d*)%",
            r"throughput.*?(\d+\.?\d*)\s*rps",
            r"p95.*?(\d+\.?\d*)\s*ms",
            r"p99.*?(\d+\.?\d*)\s*ms",
        ]

        for pattern in perf_patterns:
            matches = re.findall(pattern, full_output, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        value = float(match)
                        # Check thresholds
                        if "success_rate" in pattern and value < 95:
                            self.issues["performance_issues"].append(
                                f"Low success rate: {value}% (threshold: 95%)",
                            )
                        elif "p95" in pattern and value > 5000:
                            self.issues["performance_issues"].append(
                                f"High P95 latency: {value}ms (threshold: 5000ms)",
                            )
                        elif "p99" in pattern and value > 10000:
                            self.issues["performance_issues"].append(
                                f"High P99 latency: {value}ms (threshold: 10000ms)",
                            )
                    except ValueError:
                        pass

        # Extract duration
        duration_match = re.search(r"=+\s*(\d+\.?\d*)\s*seconds?\s*=+", full_output)
        if duration_match:
            self.metrics["total_duration"] = float(duration_match.group(1))

    def generate_report(self, stdout: str, stderr: str, return_code: int):
        """Generate comprehensive markdown report."""
        report = []
        report.append(f"# Load Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total Tests:** {self.metrics['total_tests']}")
        report.append(f"- **Passed:** {self.metrics['passed_tests']}")
        report.append(f"- **Failed:** {self.metrics['failed_tests']}")
        report.append(f"- **Duration:** {self.metrics['total_duration']:.2f}s")
        report.append(f"- **Exit Code:** {return_code}")
        report.append("")

        # Status
        if return_code == 0:
            report.append("✅ **Status:** ALL TESTS PASSED")
        else:
            report.append("❌ **Status:** TESTS FAILED")
        report.append("")

        # Issues by category
        for category, issues in self.issues.items():
            if issues:
                report.append(f"## {category.replace('_', ' ').title()}")
                report.append("")
                report.append(f"Found {len(issues)} issue(s):")
                report.append("")
                for i, issue in enumerate(issues, 1):
                    report.append(f"### {i}. {category.replace('_', ' ').title()}")
                    report.append("```")
                    report.append(issue)
                    report.append("```")
                    report.append("")

        # Full output
        report.append("## Full Test Output")
        report.append("")
        report.append("### STDOUT")
        report.append("```")
        report.append(stdout[-10000:] if len(stdout) > 10000 else stdout)  # Last 10k chars
        report.append("```")
        report.append("")

        if stderr:
            report.append("### STDERR")
            report.append("```")
            report.append(stderr[-10000:] if len(stderr) > 10000 else stderr)
            report.append("```")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        if self.issues["timeouts"]:
            report.append("- ⚠️ **Timeouts detected:** Increase timeout values or reduce concurrency")

        if self.issues["connection_issues"]:
            report.append("- ⚠️ **Connection issues:** Check server availability and network")

        if self.issues["performance_issues"]:
            report.append("- ⚠️ **Performance issues:** Review server resources and optimize queries")

        if self.metrics["failed_tests"] > 0:
            report.append("- ❌ **Failed tests:** Review error logs and fix issues before production")

        if not any(self.issues.values()) and return_code == 0:
            report.append("- ✅ **All tests passed:** System is performing well")

        report.append("")

        # Save report
        report_text = "\n".join(report)
        self.report_file.write_text(report_text)

        return report_text

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*80)
        print("LOAD TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.metrics['total_tests']}")
        print(f"Passed: {self.metrics['passed_tests']}")
        print(f"Failed: {self.metrics['failed_tests']}")
        print(f"Duration: {self.metrics['total_duration']:.2f}s")
        print("")

        for category, issues in self.issues.items():
            if issues:
                print(f"{category.replace('_', ' ').title()}: {len(issues)}")

        print("")
        print(f"Full report saved to: {self.report_file}")
        print("="*80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run load tests and analyze results")
    parser.add_argument("--test", help="Specific test to run")
    parser.add_argument("--full", action="store_true", help="Run full test suite (including slow tests)")
    parser.add_argument("--output-dir", default="tests/results", help="Output directory for reports")

    args = parser.parse_args()

    analyzer = LoadTestAnalyzer(output_dir=args.output_dir)

    print("Starting load tests...")
    stdout, stderr, return_code = analyzer.run_load_tests(
        test_path=args.test,
        quick=not args.full,
    )

    print("\nAnalyzing results...")
    analyzer.analyze_output(stdout, stderr)

    print("\nGenerating report...")
    report = analyzer.generate_report(stdout, stderr, return_code)

    analyzer.print_summary()

    # Exit with same code as tests
    sys.exit(return_code)


if __name__ == "__main__":
    main()
