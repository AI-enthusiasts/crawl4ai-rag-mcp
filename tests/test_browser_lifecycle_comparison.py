#!/usr/bin/env python3
"""
Browser Lifecycle Comparison Test
Tests 3 approaches to understand performance and memory characteristics.

Approach 1: Singleton with manual lifecycle (CURRENT)
Approach 2: Singleton with restart after each batch (PROPOSED HACK)
Approach 3: Context manager per batch (DOCUMENTATION RECOMMENDED)
"""

import asyncio
import time
import subprocess
import psutil
from typing import List, Dict, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# Test data
BATCH_1 = ["https://example.com", "https://example.org", "https://example.net"]
BATCH_2 = ["https://httpbin.org/html", "https://httpbin.org/robots.txt", "https://httpbin.org/ip"]
BATCH_3 = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3"
]

# Browser configuration
BROWSER_CONFIG = BrowserConfig(
    headless=True,
    verbose=False
)

# Crawler run config
RUN_CONFIG = CrawlerRunConfig(
    cache_mode="bypass"  # Don't use cache for fair comparison
)


def get_chrome_process_count() -> int:
    """Count Chrome/Chromium processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Count lines containing 'chrome' or 'chromium' (case-insensitive)
        count = sum(1 for line in result.stdout.lower().split('\n') 
                   if 'chrome' in line or 'chromium' in line)
        return count
    except Exception as e:
        print(f"Warning: Could not count Chrome processes: {e}")
        return -1


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_system_memory_mb() -> float:
    """Get total system memory usage by Chrome processes in MB."""
    try:
        total_mem = 0
        for proc in psutil.process_iter(['name', 'memory_info']):
            try:
                if proc.info['name'] and ('chrome' in proc.info['name'].lower() or 
                                         'chromium' in proc.info['name'].lower()):
                    total_mem += proc.info['memory_info'].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_mem / 1024 / 1024
    except Exception as e:
        print(f"Warning: Could not get system memory: {e}")
        return -1


async def approach_1_singleton_manual() -> Dict[str, Any]:
    """
    Approach 1: Singleton with manual lifecycle (CURRENT)
    Create once, reuse forever, never restart.
    """
    print("\n" + "=" * 60)
    print("=== Approach 1: Singleton with Manual Lifecycle (CURRENT) ===")
    print("=" * 60)
    
    metrics = {
        "approach": "Singleton Manual",
        "batches": [],
        "total_time": 0
    }
    
    crawler = AsyncWebCrawler(config=BROWSER_CONFIG)
    
    # Initial startup
    start_init = time.time()
    await crawler.start()
    init_time = time.time() - start_init
    print(f"Startup time: {init_time:.2f}s")
    
    await asyncio.sleep(1)  # Let processes stabilize
    
    total_start = time.time()
    
    # Batch 1
    batch_start = time.time()
    results1 = await crawler.arun_many(BATCH_1, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 1: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB")
    metrics["batches"].append({
        "batch": 1,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    # Batch 2
    batch_start = time.time()
    results2 = await crawler.arun_many(BATCH_2, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 2: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB")
    metrics["batches"].append({
        "batch": 2,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    # Batch 3
    batch_start = time.time()
    results3 = await crawler.arun_many(BATCH_3, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 3: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB")
    metrics["batches"].append({
        "batch": 3,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    metrics["total_time"] = time.time() - total_start
    print(f"Total time (excluding startup): {metrics['total_time']:.2f}s")
    
    # Cleanup
    await crawler.close()
    await asyncio.sleep(2)  # Wait for cleanup
    final_chrome = get_chrome_process_count()
    print(f"Final cleanup: Chrome processes after close(): {final_chrome}")
    metrics["final_chrome_processes"] = final_chrome
    
    return metrics


async def approach_2_singleton_restart() -> Dict[str, Any]:
    """
    Approach 2: Singleton with restart after each batch (PROPOSED HACK)
    Create once, restart between batches.
    """
    print("\n" + "=" * 60)
    print("=== Approach 2: Singleton with Restart (PROPOSED HACK) ===")
    print("=" * 60)
    
    metrics = {
        "approach": "Singleton Restart",
        "batches": [],
        "restarts": [],
        "total_time": 0
    }
    
    crawler = AsyncWebCrawler(config=BROWSER_CONFIG)
    
    # Initial startup
    start_init = time.time()
    await crawler.start()
    init_time = time.time() - start_init
    print(f"Startup time: {init_time:.2f}s")
    
    await asyncio.sleep(1)
    
    total_start = time.time()
    
    # Batch 1
    batch_start = time.time()
    results1 = await crawler.arun_many(BATCH_1, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 1: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB")
    metrics["batches"].append({
        "batch": 1,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    # Restart 1
    restart_start = time.time()
    await crawler.close()
    await asyncio.sleep(1)
    await crawler.start()
    restart_time = time.time() - restart_start
    await asyncio.sleep(1)
    print(f"Restart 1: {restart_time:.2f}s")
    metrics["restarts"].append(restart_time)
    
    # Batch 2
    batch_start = time.time()
    results2 = await crawler.arun_many(BATCH_2, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 2: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB")
    metrics["batches"].append({
        "batch": 2,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    # Restart 2
    restart_start = time.time()
    await crawler.close()
    await asyncio.sleep(1)
    await crawler.start()
    restart_time = time.time() - restart_start
    await asyncio.sleep(1)
    print(f"Restart 2: {restart_time:.2f}s")
    metrics["restarts"].append(restart_time)
    
    # Batch 3
    batch_start = time.time()
    results3 = await crawler.arun_many(BATCH_3, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 3: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB")
    metrics["batches"].append({
        "batch": 3,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    metrics["total_time"] = time.time() - total_start
    print(f"Total time (including restarts): {metrics['total_time']:.2f}s")
    
    # Cleanup
    await crawler.close()
    await asyncio.sleep(2)
    final_chrome = get_chrome_process_count()
    print(f"Final cleanup: Chrome processes after close(): {final_chrome}")
    metrics["final_chrome_processes"] = final_chrome
    
    return metrics


async def approach_3_context_manager() -> Dict[str, Any]:
    """
    Approach 3: Context manager per batch (DOCUMENTATION RECOMMENDED)
    Create new crawler for each batch.
    """
    print("\n" + "=" * 60)
    print("=== Approach 3: Context Manager per Batch (RECOMMENDED) ===")
    print("=" * 60)
    
    metrics = {
        "approach": "Context Manager",
        "batches": [],
        "total_time": 0
    }
    
    total_start = time.time()
    
    # Batch 1
    batch_start = time.time()
    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        results1 = await crawler.arun_many(BATCH_1, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 1: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB, cleanup: {chrome_count} processes")
    metrics["batches"].append({
        "batch": 1,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    # Batch 2
    batch_start = time.time()
    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        results2 = await crawler.arun_many(BATCH_2, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 2: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB, cleanup: {chrome_count} processes")
    metrics["batches"].append({
        "batch": 2,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    # Batch 3
    batch_start = time.time()
    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        results3 = await crawler.arun_many(BATCH_3, config=RUN_CONFIG)
    batch_time = time.time() - batch_start
    await asyncio.sleep(1)
    chrome_count = get_chrome_process_count()
    memory = get_system_memory_mb()
    print(f"Batch 3: {batch_time:.2f}s, Chrome processes: {chrome_count}, Memory: {memory:.0f}MB, cleanup: {chrome_count} processes")
    metrics["batches"].append({
        "batch": 3,
        "time": batch_time,
        "chrome_processes": chrome_count,
        "memory_mb": memory
    })
    
    metrics["total_time"] = time.time() - total_start
    print(f"Total time: {metrics['total_time']:.2f}s")
    
    await asyncio.sleep(2)
    final_chrome = get_chrome_process_count()
    print(f"Final state: {final_chrome} Chrome processes")
    metrics["final_chrome_processes"] = final_chrome
    
    return metrics


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print comparison table of all approaches."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"\n{'Approach':<25} {'Batch 1':<15} {'Batch 2':<15} {'Batch 3':<15} {'Total':<10}")
    print("-" * 80)
    
    for result in results:
        approach = result["approach"]
        batch_times = [b["time"] for b in result["batches"]]
        total = result["total_time"]
        
        print(f"{approach:<25} {batch_times[0]:>6.2f}s {'':<8} "
              f"{batch_times[1]:>6.2f}s {'':<8} "
              f"{batch_times[2]:>6.2f}s {'':<8} "
              f"{total:>6.2f}s")
    
    print("\n" + "=" * 80)
    print("MEMORY & PROCESS ANALYSIS")
    print("=" * 80)
    
    # Header
    print(f"\n{'Approach':<25} {'Chrome Procs (B1/B2/B3)':<30} {'Memory (B1/B2/B3)':<30} {'Final':<10}")
    print("-" * 80)
    
    for result in results:
        approach = result["approach"]
        chrome_counts = [b["chrome_processes"] for b in result["batches"]]
        memories = [b["memory_mb"] for b in result["batches"]]
        final_chrome = result["final_chrome_processes"]
        
        chrome_str = f"{chrome_counts[0]}/{chrome_counts[1]}/{chrome_counts[2]}"
        memory_str = f"{memories[0]:.0f}/{memories[1]:.0f}/{memories[2]:.0f}MB"
        
        print(f"{approach:<25} {chrome_str:<30} {memory_str:<30} {final_chrome:<10}")


def analyze_results(results: List[Dict[str, Any]]):
    """Provide analysis and recommendation based on results."""
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    singleton_manual = results[0]
    singleton_restart = results[1]
    context_manager = results[2]
    
    # Memory leak analysis
    print("\n1. MEMORY LEAK DETECTION:")
    print("-" * 80)
    
    for result in results:
        chrome_counts = [b["chrome_processes"] for b in result["batches"]]
        final = result["final_chrome_processes"]
        
        if chrome_counts[0] < chrome_counts[1] < chrome_counts[2]:
            print(f"❌ {result['approach']}: LEAK DETECTED - processes growing ({chrome_counts[0]} → {chrome_counts[1]} → {chrome_counts[2]})")
        elif all(c == chrome_counts[0] for c in chrome_counts):
            print(f"✅ {result['approach']}: NO LEAK - stable at {chrome_counts[0]} processes")
        else:
            print(f"⚠️  {result['approach']}: UNCLEAR - processes: {chrome_counts}")
        
        if final > 5:
            print(f"   ❌ Cleanup issue: {final} processes remain after close()")
        else:
            print(f"   ✅ Good cleanup: {final} processes after close()")
    
    # Performance comparison
    print("\n2. PERFORMANCE COMPARISON:")
    print("-" * 80)
    
    fastest = min(results, key=lambda r: r["total_time"])
    slowest = max(results, key=lambda r: r["total_time"])
    
    print(f"Fastest: {fastest['approach']} ({fastest['total_time']:.2f}s)")
    print(f"Slowest: {slowest['approach']} ({slowest['total_time']:.2f}s)")
    print(f"Overhead: {slowest['total_time'] - fastest['total_time']:.2f}s ({((slowest['total_time'] / fastest['total_time']) - 1) * 100:.1f}% slower)")
    
    # Batch time stability
    print("\n3. BATCH TIME STABILITY:")
    print("-" * 80)
    
    for result in results:
        batch_times = [b["time"] for b in result["batches"]]
        avg_time = sum(batch_times) / len(batch_times)
        variance = sum((t - avg_time) ** 2 for t in batch_times) / len(batch_times)
        std_dev = variance ** 0.5
        
        stability = "stable" if std_dev < 0.5 else "unstable"
        print(f"{result['approach']}: avg={avg_time:.2f}s, σ={std_dev:.2f}s ({stability})")
    
    # Recommendation
    print("\n4. RECOMMENDATION:")
    print("-" * 80)
    
    # Check if singleton manual has memory leak
    singleton_chrome = [b["chrome_processes"] for b in singleton_manual["batches"]]
    has_leak = singleton_chrome[0] < singleton_chrome[1] < singleton_chrome[2]
    
    if has_leak:
        print("❌ CURRENT APPROACH (Singleton Manual) HAS MEMORY LEAK")
        print("   Processes grow with each batch, leading to resource exhaustion.")
        print()
        print("✅ RECOMMENDED: Switch to Context Manager (Approach 3)")
        print("   Reasons:")
        print("   - No memory leak (stable process count)")
        print("   - Automatic cleanup (no manual close() needed)")
        print("   - Moderate performance (~2x slower but acceptable)")
        print("   - Follows library best practices")
        print()
        print("⚠️  ALTERNATIVE: Singleton with Restart (Approach 2)")
        print("   - Fixes memory leak by restarting")
        print("   - But slowest due to restart overhead")
        print("   - Only use if context manager doesn't work for your use case")
    else:
        print("✅ CURRENT APPROACH (Singleton Manual) IS OPTIMAL")
        print("   No memory leak detected, fastest performance.")
        print("   BUT: Investigate why production has 366 processes - likely different issue.")


async def main():
    """Run all tests and provide analysis."""
    print("=" * 80)
    print("BROWSER LIFECYCLE COMPARISON TEST")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial Chrome processes: {get_chrome_process_count()}")
    print(f"Initial system memory: {get_system_memory_mb():.0f}MB")
    
    results = []
    
    try:
        # Run all three approaches
        result1 = await approach_1_singleton_manual()
        results.append(result1)
        
        await asyncio.sleep(3)  # Let system stabilize between tests
        
        result2 = await approach_2_singleton_restart()
        results.append(result2)
        
        await asyncio.sleep(3)
        
        result3 = await approach_3_context_manager()
        results.append(result3)
        
        # Print comparison and analysis
        print_comparison_table(results)
        analyze_results(results)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "=" * 80)
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Final Chrome processes: {get_chrome_process_count()}")
        print(f"Final system memory: {get_system_memory_mb():.0f}MB")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
