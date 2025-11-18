#!/usr/bin/env python3
"""
Test script to compare browser lifecycle management approaches in Crawl4AI.

Compares:
1. Singleton crawler (reuse same instance across batches)
2. Context manager per batch (create/destroy for each batch)

Measures:
- Total execution time
- Memory usage after each batch
- Number of Chrome processes
- Peak memory usage
"""

import asyncio
import gc
import subprocess
import time
from typing import Any

import psutil
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

# Test configuration
NUM_URLS = 5
NUM_BATCHES = 3
TEST_URL = "https://httpbin.org/delay/1"  # Consistent 1-second delay

# Browser configuration
BROWSER_CONFIG = BrowserConfig(
    headless=True,
    verbose=False,
)

CRAWLER_CONFIG = CrawlerRunConfig(
    cache_mode=CacheMode.DISABLED,  # Disable caching for consistent results
)


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def count_chrome_processes() -> int:
    """Count Chrome/Chromium processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Count lines containing chrome/chromium
        count = sum(
            1
            for line in result.stdout.split("\n")
            if "chrome" in line.lower() or "chromium" in line.lower()
        )
        return count
    except Exception as e:
        print(f"Warning: Could not count Chrome processes: {e}")
        return -1


async def test_singleton_approach() -> dict[str, Any]:
    """Test Approach 1: Singleton crawler reused across batches."""
    print("\n" + "=" * 60)
    print("Testing Approach 1: Singleton Crawler")
    print("=" * 60)

    urls = [TEST_URL] * NUM_URLS
    results: dict[str, Any] = {
        "approach": "Singleton",
        "memory_per_batch": [],
        "chrome_processes": 0,
        "peak_memory": 0,
        "total_time": 0,
    }

    # Force garbage collection before starting
    gc.collect()
    start_memory = get_memory_mb()
    print(f"Starting memory: {start_memory:.2f} MB")

    start_time = time.time()

    # Create and start crawler once
    crawler = AsyncWebCrawler(config=BROWSER_CONFIG)
    await crawler.start()
    print("Crawler started")

    try:
        # Run multiple batches with same crawler
        for batch_num in range(1, NUM_BATCHES + 1):
            print(f"\nBatch {batch_num}/{NUM_BATCHES}...")
            batch_start = time.time()

            await crawler.arun_many(urls, config=CRAWLER_CONFIG)

            batch_time = time.time() - batch_start
            current_memory = get_memory_mb()
            results["memory_per_batch"].append(current_memory)
            results["peak_memory"] = max(results["peak_memory"], current_memory)

            print(f"  Batch {batch_num} completed in {batch_time:.2f}s")
            print(f"  Memory after batch: {current_memory:.2f} MB")

    finally:
        # Close crawler
        await crawler.close()
        print("\nCrawler closed")

    results["total_time"] = time.time() - start_time

    # Give time for processes to clean up
    await asyncio.sleep(2)
    gc.collect()

    # Count Chrome processes after cleanup
    results["chrome_processes"] = count_chrome_processes()

    print(f"\nTotal execution time: {results['total_time']:.2f}s")
    print(f"Chrome processes remaining: {results['chrome_processes']}")
    print(f"Peak memory: {results['peak_memory']:.2f} MB")

    return results


async def test_context_manager_approach() -> dict[str, Any]:
    """Test Approach 2: Context manager per batch."""
    print("\n" + "=" * 60)
    print("Testing Approach 2: Context Manager Per Batch")
    print("=" * 60)

    urls = [TEST_URL] * NUM_URLS
    results: dict[str, Any] = {
        "approach": "Context Manager",
        "memory_per_batch": [],
        "chrome_processes": 0,
        "peak_memory": 0,
        "total_time": 0,
    }

    # Force garbage collection before starting
    gc.collect()
    start_memory = get_memory_mb()
    print(f"Starting memory: {start_memory:.2f} MB")

    start_time = time.time()

    # Create new crawler for each batch
    for batch_num in range(1, NUM_BATCHES + 1):
        print(f"\nBatch {batch_num}/{NUM_BATCHES}...")
        batch_start = time.time()

        async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
            await crawler.arun_many(urls, config=CRAWLER_CONFIG)

        batch_time = time.time() - batch_start
        current_memory = get_memory_mb()
        results["memory_per_batch"].append(current_memory)
        results["peak_memory"] = max(results["peak_memory"], current_memory)

        print(f"  Batch {batch_num} completed in {batch_time:.2f}s")
        print(f"  Memory after batch: {current_memory:.2f} MB")

        # Force garbage collection between batches
        gc.collect()
        await asyncio.sleep(1)

    results["total_time"] = time.time() - start_time

    # Give time for processes to clean up
    await asyncio.sleep(2)
    gc.collect()

    # Count Chrome processes after cleanup
    results["chrome_processes"] = count_chrome_processes()

    print(f"\nTotal execution time: {results['total_time']:.2f}s")
    print(f"Chrome processes remaining: {results['chrome_processes']}")
    print(f"Peak memory: {results['peak_memory']:.2f} MB")

    return results


def print_comparison(results1: dict[str, Any], results2: dict[str, Any]) -> None:
    """Print comparison between both approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nApproach 1 ({results1['approach']}):")
    print(f"  - Total time: {results1['total_time']:.2f} seconds")
    for i, mem in enumerate(results1["memory_per_batch"], 1):
        print(f"  - Memory after batch {i}: {mem:.2f} MB")
    print(f"  - Chrome processes after batch {NUM_BATCHES}: {results1['chrome_processes']}")
    print(f"  - Peak memory: {results1['peak_memory']:.2f} MB")

    print(f"\nApproach 2 ({results2['approach']}):")
    print(f"  - Total time: {results2['total_time']:.2f} seconds")
    for i, mem in enumerate(results2["memory_per_batch"], 1):
        print(f"  - Memory after batch {i}: {mem:.2f} MB")
    print(f"  - Chrome processes after batch {NUM_BATCHES}: {results2['chrome_processes']}")
    print(f"  - Peak memory: {results2['peak_memory']:.2f} MB")

    # Calculate differences
    time_diff = results2["total_time"] - results1["total_time"]
    time_diff_pct = (time_diff / results1["total_time"]) * 100
    mem_diff = results2["peak_memory"] - results1["peak_memory"]
    mem_diff_pct = (mem_diff / results1["peak_memory"]) * 100

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    print(f"\nTime difference: {time_diff:+.2f}s ({time_diff_pct:+.1f}%)")
    print(f"Peak memory difference: {mem_diff:+.2f} MB ({mem_diff_pct:+.1f}%)")

    print("\nConclusion:")
    if abs(time_diff_pct) < 5 and abs(mem_diff_pct) < 5:
        print("  Both approaches show similar performance (< 5% difference).")
        print("  Choice depends on use case:")
        print("    - Singleton: Better for continuous crawling (slightly lower overhead)")
        print("    - Context Manager: Better for isolation, error recovery, batch jobs")
    elif time_diff_pct < -10:
        print(f"  Approach 1 (Singleton) is significantly faster ({abs(time_diff_pct):.1f}% faster)")
        print("  Reason: Avoids browser startup/shutdown overhead between batches")
    elif time_diff_pct > 10:
        print(f"  Approach 2 (Context Manager) is significantly faster ({time_diff_pct:.1f}% faster)")
        print("  Reason: Fresh browser state may avoid memory buildup")
    else:
        print("  Performance difference is minimal.")

    if mem_diff_pct < -10:
        print(f"  Approach 1 (Singleton) uses less memory ({abs(mem_diff_pct):.1f}% less)")
    elif mem_diff_pct > 10:
        print(f"  Approach 2 (Context Manager) uses less memory ({mem_diff_pct:.1f}% less)")

    # Memory trend analysis
    print("\nMemory trends:")
    mem1_growth = results1["memory_per_batch"][-1] - results1["memory_per_batch"][0]
    mem2_growth = results2["memory_per_batch"][-1] - results2["memory_per_batch"][0]
    print(f"  - Singleton memory growth: {mem1_growth:+.2f} MB")
    print(f"  - Context Manager memory growth: {mem2_growth:+.2f} MB")

    if abs(mem1_growth) > 50:
        print("  ⚠ WARNING: Singleton shows significant memory growth (potential leak)")
    if abs(mem2_growth) > 50:
        print("  ⚠ WARNING: Context Manager shows significant memory growth (potential leak)")


async def main() -> None:
    """Run both tests and compare results."""
    print("Browser Lifecycle Management Comparison Test")
    print(f"Configuration: {NUM_URLS} URLs × {NUM_BATCHES} batches")
    print(f"Test URL: {TEST_URL}")

    # Run tests sequentially to avoid interference
    results1 = await test_singleton_approach()

    # Wait between tests to ensure clean state
    print("\n" + "=" * 60)
    print("Waiting 5 seconds before next test...")
    print("=" * 60)
    await asyncio.sleep(5)
    gc.collect()

    results2 = await test_context_manager_approach()

    # Print comparison
    print_comparison(results1, results2)


if __name__ == "__main__":
    asyncio.run(main())
