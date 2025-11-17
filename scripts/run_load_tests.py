#!/usr/bin/env python3
"""
Load testing runner for Crawl4AI MCP Server.

This script provides a convenient interface for running load tests
with service status checking, result reporting, and recommendations.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install 'rich' for better output: uv add rich")


class LoadTestRunner:
    """Runner for MCP load tests."""

    def __init__(self, verbose: bool = False):
        """Initialize runner.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE else None
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "tests" / "results" / "load_tests"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def print(self, message: str, style: Optional[str] = None):
        """Print message with optional styling.

        Args:
            message: Message to print
            style: Rich style (e.g., 'green', 'yellow', 'red')
        """
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)

    def check_docker_container(self, pattern: str) -> Optional[str]:
        """Check if Docker container matching pattern is running.

        Args:
            pattern: Pattern to match container name

        Returns:
            Container name if found, None otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return None

            containers = result.stdout.strip().split("\n")
            for container in containers:
                if container.lower().startswith(pattern.lower()):
                    return container
            return None
        except FileNotFoundError:
            return None

    def check_process(self, name: str) -> Optional[int]:
        """Check if process is running.

        Args:
            name: Process name to search for

        Returns:
            Process PID if found, None otherwise
        """
        try:
            result = subprocess.run(
                ["pgrep", "-f", name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split("\n")[0])
            return None
        except (FileNotFoundError, ValueError):
            return None

    def check_systemd_service(self, name: str) -> bool:
        """Check if systemd service is active.

        Args:
            name: Service name

        Returns:
            True if service is active, False otherwise
        """
        for cmd in [
            ["systemctl", "--user", "is-active", name],
            ["systemctl", "is-active", name],
        ]:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return True
            except FileNotFoundError:
                continue
        return False

    def check_services(self) -> Dict[str, any]:
        """Check status of required services.

        Returns:
            Dictionary with service status information
        """
        status = {}

        # Check Crawl4AI container
        crawl4ai = self.check_docker_container("crawl4ai") or \
                   self.check_docker_container("mcp-crawl4ai")
        status["crawl4ai"] = {
            "running": crawl4ai is not None,
            "name": crawl4ai,
        }

        # Check mcpproxy
        mcpproxy_systemd = self.check_systemd_service("mcpproxy")
        mcpproxy_process = self.check_process("mcpproxy")

        status["mcpproxy"] = {
            "running": mcpproxy_systemd or mcpproxy_process is not None,
            "type": "systemd" if mcpproxy_systemd else "process" if mcpproxy_process else None,
            "pid": mcpproxy_process,
        }

        return status

    def print_service_status(self, status: Dict[str, any]):
        """Print service status.

        Args:
            status: Service status dictionary
        """
        if self.console:
            self.console.print("\n[yellow]Checking service status...[/yellow]")
        else:
            print("\nChecking service status...")

        # Crawl4AI status
        if status["crawl4ai"]["running"]:
            msg = f"✓ Crawl4AI container running: {status['crawl4ai']['name']}"
            self.print(msg, "green")
        else:
            self.print("⚠ Crawl4AI container not found", "yellow")

        # mcpproxy status
        if status["mcpproxy"]["running"]:
            if status["mcpproxy"]["type"] == "systemd":
                msg = "✓ mcpproxy service is active (systemd)"
            else:
                msg = f"✓ mcpproxy process running (PID: {status['mcpproxy']['pid']})"
            self.print(msg, "green")
        else:
            self.print("⚠ mcpproxy not found", "yellow")

        # Overall status
        all_running = status["crawl4ai"]["running"] and status["mcpproxy"]["running"]
        if all_running:
            self.print("✓ All services running - can test against real MCP server", "green")
            self.print("ℹ  Currently using mock implementation", "blue")
            self.print("ℹ  To enable real MCP testing, update load_tester fixture", "blue")
        else:
            self.print("ℹ  Using mock implementation (services not required)", "blue")
            if not status["crawl4ai"]["running"]:
                self.print("  - Start crawl4ai: docker-compose up -d", "yellow")
            if not status["mcpproxy"]["running"]:
                self.print("  - Start mcpproxy: systemctl --user start mcpproxy", "yellow")

        print()

    def get_test_targets(self, mode: str) -> List[str]:
        """Get test targets for specified mode.

        Args:
            mode: Test mode (quick, stress, endurance, full)

        Returns:
            List of test target paths
        """
        test_file = "tests/integration/test_mcp_load_testing.py"

        targets = {
            "quick": ["TestMCPThroughput", "TestMCPLatency"],
            "stress": ["TestMCPStress"],
            "endurance": ["TestMCPEndurance"],
            "concurrency": ["TestMCPConcurrency"],
            "full": [
                "TestMCPThroughput",
                "TestMCPLatency",
                "TestMCPConcurrency",
                "TestMCPStress",
                "TestMCPEndurance",
            ],
        }

        return [f"{test_file}::{target}" for target in targets.get(mode, targets["full"])]

    def run_tests(self, mode: str, report_file: Path) -> int:
        """Run load tests.

        Args:
            mode: Test mode
            report_file: Path to save JSON report

        Returns:
            Exit code from pytest
        """
        test_targets = self.get_test_targets(mode)

        # Build pytest command
        cmd = ["uv", "run", "pytest"] + test_targets + [
            "-v",
            "--json-report",
            f"--json-report-file={report_file}",
        ]

        if self.verbose:
            cmd.append("-s")

        # Print test info
        mode_names = {
            "quick": "quick load tests (throughput and latency)",
            "stress": "stress tests",
            "endurance": "endurance tests",
            "concurrency": "concurrency tests",
            "full": "full load test suite",
        }

        self.print(f"Running {mode_names.get(mode, mode)} tests...", "yellow")
        self.print(f"Results will be saved to: {report_file}", "blue")
        print()

        # Run tests
        result = subprocess.run(cmd, cwd=self.project_root)

        return result.returncode

    def print_summary(self, report_file: Path, exit_code: int):
        """Print test summary.

        Args:
            report_file: Path to JSON report
            exit_code: Test exit code
        """
        print()
        if self.console:
            self.console.print(Panel.fit(
                "[bold]Test Summary[/bold]",
                border_style="blue",
            ))
        else:
            print("=" * 60)
            print("Test Summary")
            print("=" * 60)

        # Load report
        if not report_file.exists():
            self.print("✗ No test results found", "red")
            return

        with report_file.open() as f:
            data = json.load(f)

        summary = data.get("summary", {})
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        duration = data.get("duration", 0)

        # Print summary
        if self.console:
            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value")

            table.add_row("Total Tests:", f"[blue]{total}[/blue]")
            table.add_row("Passed:", f"[green]{passed}[/green]")
            table.add_row("Failed:", f"[red]{failed}[/red]")
            table.add_row("Duration:", f"[blue]{duration:.2f}s[/blue]")

            self.console.print(table)
        else:
            print(f"Total Tests:    {total}")
            print(f"Passed:         {passed}")
            print(f"Failed:         {failed}")
            print(f"Duration:       {duration:.2f}s")

        print()
        self.print(f"✓ Results saved to: {report_file}", "green")

    def print_recommendations(self, exit_code: int):
        """Print recommendations based on test results.

        Args:
            exit_code: Test exit code
        """
        print()
        if self.console:
            self.console.print(Panel.fit(
                "[bold]Recommendations[/bold]",
                border_style="blue",
            ))
        else:
            print("=" * 60)
            print("Recommendations")
            print("=" * 60)

        if exit_code == 0:
            self.print("✓ All load tests passed!", "green")
            print()
            print("Next steps:")
            print("  • Review detailed metrics in the JSON report")
            print("  • Compare with previous test runs")
            print("  • Consider increasing load for stress testing")
        else:
            self.print("✗ Some load tests failed", "red")
            print()
            print("Troubleshooting steps:")
            print("  1. Review failed test output above")
            print("  2. Check if services are running properly")
            print("  3. Verify test thresholds are appropriate")
            print("  4. Consider reducing concurrency level")
            print("  5. Check system resources (memory, CPU)")

        print()
        self.print("For more information, see: tests/integration/LOAD_TESTING_README.md", "blue")
        print()

    def run(self, mode: str) -> int:
        """Run complete load testing workflow.

        Args:
            mode: Test mode

        Returns:
            Exit code
        """
        # Print header
        if self.console:
            self.console.print(Panel.fit(
                "[bold blue]Crawl4AI MCP Server Load Testing Suite[/bold blue]",
                border_style="blue",
            ))
        else:
            print("=" * 60)
            print("Crawl4AI MCP Server Load Testing Suite")
            print("=" * 60)

        # Check services
        status = self.check_services()
        self.print_service_status(status)

        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"load_test_report_{timestamp}.json"

        # Run tests
        exit_code = self.run_tests(mode, report_file)

        # Print summary and recommendations
        self.print_summary(report_file, exit_code)
        self.print_recommendations(exit_code)

        return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run load tests for Crawl4AI MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                  # Quick performance check
  %(prog)s --full                   # Complete load test suite
  %(prog)s --stress                 # Find system limits
  %(prog)s --quick --verbose        # Quick tests with detailed output
        """,
    )

    # Test mode options (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Run quick load tests (throughput and latency only)",
    )
    mode_group.add_argument(
        "-f", "--full",
        action="store_true",
        help="Run full load test suite (default)",
    )
    mode_group.add_argument(
        "-s", "--stress",
        action="store_true",
        help="Run stress tests only",
    )
    mode_group.add_argument(
        "-e", "--endurance",
        action="store_true",
        help="Run endurance tests only",
    )
    mode_group.add_argument(
        "-c", "--concurrency",
        action="store_true",
        help="Run concurrency tests only",
    )

    # Other options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show test output)",
    )

    args = parser.parse_args()

    # Determine mode
    if args.quick:
        mode = "quick"
    elif args.stress:
        mode = "stress"
    elif args.endurance:
        mode = "endurance"
    elif args.concurrency:
        mode = "concurrency"
    else:
        mode = "full"

    # Run tests
    runner = LoadTestRunner(verbose=args.verbose)
    exit_code = runner.run(mode)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
