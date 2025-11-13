# Crawl4AI Monitoring & Observability Capabilities

**Date:** 2025-10-28  
**Version:** Crawl4AI v0.7.x  
**Author:** Research Analysis  
**Source:** Official Crawl4AI Documentation (https://docs.crawl4ai.com/)

---

## 📋 Executive Summary

This document provides a comprehensive analysis of monitoring and observability capabilities in Crawl4AI library. Crawl4AI provides **programmatic building blocks** for monitoring but does not include a built-in monitoring dashboard or external integrations (Prometheus, OpenTelemetry, etc.). All metrics are accessible through Python API.

### Key Findings

✅ **Available:**
- Network request/response capture
- Browser console message interception
- Resource usage monitoring (memory, CPU)
- Task progress tracking
- Performance metrics collection
- Error tracking and diagnostics

❌ **Not Available:**
- Built-in Prometheus exporter
- OpenTelemetry integration
- Web-based monitoring dashboard
- Alert/webhook system
- Distributed tracing

---

## 🔍 1. Network & Console Capture

**Documentation:** [Network & Console Capture](https://docs.crawl4ai.com/advanced/network-console-capture/)

### Overview

Crawl4AI can capture all network traffic and browser console messages during crawling operations, providing deep visibility into page behavior.

### Capabilities

#### 1.1 Network Request Monitoring

Captures three types of network events:

1. **Request Events** - Outgoing HTTP requests
2. **Response Events** - Server responses with timing data
3. **Failed Request Events** - Network failures and errors

**Data Structure:**

```python
# Request Event
{
    "event_type": "request",
    "url": "https://api.example.com/data",
    "method": "GET",
    "headers": {"User-Agent": "...", "Accept": "..."},
    "post_data": "key=value&otherkey=value",  # For POST/PUT
    "resource_type": "fetch",  # document, stylesheet, image, script, etc.
    "is_navigation_request": false,
    "timestamp": 1633456789.123
}

# Response Event
{
    "event_type": "response",
    "url": "https://api.example.com/data",
    "status": 200,
    "status_text": "OK",
    "headers": {"Content-Type": "application/json", "Cache-Control": "..."},
    "from_service_worker": false,
    "request_timing": {
        "requestTime": 1234.56,
        "receiveHeadersEnd": 1234.78
    },
    "timestamp": 1633456789.456
}

# Failed Request Event
{
    "event_type": "request_failed",
    "url": "https://example.com/missing.png",
    "method": "GET",
    "resource_type": "image",
    "failure_text": "net::ERR_ABORTED 404",
    "timestamp": 1633456789.789
}
```

#### 1.2 Console Message Monitoring

Captures all browser console output:

```python
{
    "type": "error",  # log, error, warning, info, debug
    "text": "Uncaught TypeError: Cannot read property 'length' of undefined",
    "location": "https://example.com/script.js:123:45",
    "timestamp": 1633456790.123
}
```

### Implementation

#### Enable Monitoring

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

config = CrawlerRunConfig(
    capture_network_requests=True,   # Enable network capture
    capture_console_messages=True    # Enable console capture
)

async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(url="https://example.com", config=config)
    
    # Access captured data
    network_events = result.network_requests      # List[Dict[str, Any]]
    console_messages = result.console_messages    # List[Dict[str, Any]]
```

#### Analysis Examples

**Count Request Types:**

```python
if result.network_requests:
    requests = [r for r in result.network_requests if r.get("event_type") == "request"]
    responses = [r for r in result.network_requests if r.get("event_type") == "response"]
    failures = [r for r in result.network_requests if r.get("event_type") == "request_failed"]
    
    print(f"Requests: {len(requests)}")
    print(f"Responses: {len(responses)}")
    print(f"Failures: {len(failures)}")
```

**Find API Calls:**

```python
api_calls = [
    r for r in result.network_requests 
    if r.get("event_type") == "request" and "api" in r.get("url", "")
]

for call in api_calls:
    print(f"{call['method']} {call['url']}")
```

**Analyze Console Errors:**

```python
if result.console_messages:
    # Group by message type
    message_types = {}
    for msg in result.console_messages:
        msg_type = msg.get("type", "unknown")
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    
    print(f"Message types: {message_types}")
    
    # Show errors
    errors = [msg for msg in result.console_messages if msg.get("type") == "error"]
    for err in errors:
        print(f"Error: {err.get('text')}")
        print(f"Location: {err.get('location')}")
```

**Export for Analysis:**

```python
import json

# Save all monitoring data
monitoring_data = {
    "url": result.url,
    "timestamp": time.time(),
    "network_requests": result.network_requests or [],
    "console_messages": result.console_messages or [],
    "status_code": result.status_code,
    "success": result.success
}

with open("monitoring_data.json", "w") as f:
    json.dump(monitoring_data, f, indent=2)
```

### Use Cases

1. **API Discovery** - Identify hidden endpoints in SPAs
2. **Debugging** - Track JavaScript errors affecting functionality
3. **Security Auditing** - Detect unwanted third-party requests
4. **Performance Analysis** - Identify slow-loading resources
5. **Ad/Tracker Detection** - Catalog advertising and tracking calls

### Performance Impact

- Minimal overhead (~2-5% slower)
- Memory usage increases proportionally to request count
- Recommended for debugging/analysis, not production high-volume crawling

---

## 📊 2. CrawlerMonitor Component

**Example:** [crawler_monitor_example.py](https://github.com/unclecode/crawl4ai/blob/main/docs/examples/crawler_monitor_example.py)

### Overview

`CrawlerMonitor` provides real-time visualization and tracking of crawler operations, including task states, queue statistics, and memory pressure monitoring.

### Features

- ✅ Real-time task status tracking
- ✅ Memory pressure monitoring (NORMAL/PRESSURE/CRITICAL)
- ✅ Queue statistics (queued/active/completed/failed)
- ✅ Wait time and processing time tracking
- ✅ Thread-safe concurrent updates

### Task States

```python
from crawl4ai.models import CrawlStatus

CrawlStatus.QUEUED        # Task waiting in queue
CrawlStatus.IN_PROGRESS   # Task being processed
CrawlStatus.COMPLETED     # Task finished successfully
CrawlStatus.FAILED        # Task failed with error
```

### Implementation

```python
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.models import CrawlStatus
import time

# Initialize monitor
monitor = CrawlerMonitor()

# Add task to queue
task_id = "task-001"
url = "https://example.com"
monitor.add_task(task_id, url)

# Update to in-progress
monitor.update_task(
    task_id=task_id,
    status=CrawlStatus.IN_PROGRESS,
    start_time=time.time(),
    wait_time=2.5  # seconds in queue
)

# Update memory status
monitor.update_memory_status("PRESSURE")  # NORMAL, PRESSURE, CRITICAL

# Update queue stats
monitor.update_queue_stats(
    queued=5,
    active=3,
    completed=42,
    failed=2
)

# Complete task
monitor.update_task(
    task_id=task_id,
    status=CrawlStatus.COMPLETED,
    memory_mb=85.3,
    process_time=8.2
)
```

### Memory Status Levels

| Level | Description | Recommended Action |
|-------|-------------|-------------------|
| `NORMAL` | Memory usage < 70% | Continue normal operations |
| `PRESSURE` | Memory usage 70-85% | Reduce concurrency, slow down |
| `CRITICAL` | Memory usage > 85% | Pause new tasks, wait for cleanup |

### Integration Example

```python
import threading
import uuid
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.models import CrawlStatus

def process_url(monitor, url):
    """Process a single URL with monitoring."""
    task_id = str(uuid.uuid4())
    
    # Track task creation
    monitor.add_task(task_id, url)
    queue_time = time.time()
    
    # ... wait for resources ...
    
    # Start processing
    wait_duration = time.time() - queue_time
    monitor.update_task(
        task_id=task_id,
        status=CrawlStatus.IN_PROGRESS,
        start_time=time.time(),
        wait_time=wait_duration
    )
    
    try:
        # Actual crawling work here
        result = crawl_url(url)
        
        # Mark completed
        monitor.update_task(
            task_id=task_id,
            status=CrawlStatus.COMPLETED,
            memory_mb=get_memory_usage(),
            process_time=time.time() - start_time
        )
        
    except Exception as e:
        monitor.update_task(
            task_id=task_id,
            status=CrawlStatus.FAILED,
            error_message=str(e)
        )

# Process multiple URLs concurrently
monitor = CrawlerMonitor()
threads = []

for url in urls:
    thread = threading.Thread(target=process_url, args=(monitor, url))
    thread.start()
    threads.append(thread)

# Monitor updates queue stats automatically
monitor.update_queue_stats(
    queued=len([t for t in threads if not t.is_alive()]),
    active=len([t for t in threads if t.is_alive()]),
    completed=completed_count,
    failed=failed_count
)
```

---

## ⚙️ 3. Resource-Aware Crawling

**Documentation:** [arun_many() - Multi-URL Crawling](https://docs.crawl4ai.com/api/async-webcrawler/#4-batch-processing-arun_many)

### Overview

The `arun_many()` method includes intelligent resource monitoring and adaptive rate limiting for batch crawling operations.

### Key Features

#### 3.1 Memory Monitoring

```python
from crawl4ai import CrawlerRunConfig

config = CrawlerRunConfig(
    memory_threshold_percent=70.0,  # Pause when memory exceeds 70%
    check_interval=1.0,              # Check every 1 second
    max_session_permit=20            # Maximum concurrent sessions
)
```

**How it works:**
- Monitors system memory usage every `check_interval` seconds
- Pauses new tasks when memory exceeds `memory_threshold_percent`
- Resumes when memory drops below threshold
- Prevents system overload and OOM crashes

#### 3.2 Rate Limiting

```python
from crawl4ai import RateLimitConfig

rate_config = RateLimitConfig(
    base_delay=1.0,           # Minimum delay between requests (seconds)
    max_delay=60.0,           # Maximum delay after rate limit detection
    max_retries=3,            # Number of retry attempts
    rate_limit_codes=[429],   # HTTP codes that indicate rate limiting
)

config = CrawlerRunConfig(
    enable_rate_limiting=True,
    rate_limit_config=rate_config
)
```

**Features:**
- Automatic delay between requests
- Exponential backoff on rate limit detection
- Domain-specific rate limiting
- Configurable retry strategy

#### 3.3 Progress Monitoring

```python
from crawl4ai import DisplayMode

config = CrawlerRunConfig(
    display_mode=DisplayMode.DETAILED  # DETAILED, BRIEF, or None
)

async with AsyncWebCrawler() as crawler:
    results = await crawler.arun_many(urls, config=config)
    
    # Progress shown automatically:
    # [1/100] Processing: https://example.com/page1
    # [2/100] Processing: https://example.com/page2
    # Memory: 45.2 MB | Active: 5 | Completed: 2 | Failed: 0
```

### Complete Example

```python
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, DisplayMode

async def monitor_batch_crawl():
    """Crawl multiple URLs with resource monitoring."""
    
    urls = [f"https://example.com/page{i}" for i in range(100)]
    
    config = CrawlerRunConfig(
        # Rate limiting
        enable_rate_limiting=True,
        rate_limit_config=RateLimitConfig(base_delay=1.0),
        
        # Resource monitoring
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=10,
        
        # Progress display
        display_mode=DisplayMode.DETAILED,
        verbose=True
    )
    
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(urls, config=config)
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\nCompleted: {len(successful)}/{len(urls)}")
        print(f"Failed: {len(failed)}")
        
        # Check dispatch metrics
        for result in successful:
            if result.dispatch_result:
                dr = result.dispatch_result
                print(f"URL: {result.url}")
                print(f"  Memory: {dr.memory_usage:.1f} MB (Peak: {dr.peak_memory:.1f} MB)")
                print(f"  Duration: {(dr.end_time - dr.start_time).total_seconds():.2f}s")

asyncio.run(monitor_batch_crawl())
```

---

## 📦 4. Dispatch Result Metrics

**Documentation:** [CrawlResult - dispatch_result](https://docs.crawl4ai.com/api/crawl-result/#6-dispatch_result-optional)

### Overview

When using `arun_many()` with dispatchers, each `CrawlResult` includes a `dispatch_result` object with detailed performance metrics.

### Available Metrics

```python
class DispatchResult:
    task_id: str              # Unique task identifier
    memory_usage: float       # Memory used at completion (MB)
    peak_memory: float        # Peak memory during task (MB)
    start_time: datetime      # Task start timestamp
    end_time: datetime        # Task completion timestamp
    error_message: str        # Dispatcher-specific errors
```

### Usage Example

```python
async with AsyncWebCrawler() as crawler:
    results = await crawler.arun_many(urls)
    
    # Analyze performance metrics
    for result in results:
        if result.success and result.dispatch_result:
            dr = result.dispatch_result
            
            # Calculate duration
            duration = (dr.end_time - dr.start_time).total_seconds()
            
            # Log metrics
            print(f"Task {dr.task_id[:8]}:")
            print(f"  URL: {result.url}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Memory: {dr.memory_usage:.1f} MB")
            print(f"  Peak Memory: {dr.peak_memory:.1f} MB")
            print(f"  Status: {result.status_code}")
```

### Metrics Aggregation

```python
import statistics

# Collect metrics
durations = []
memory_usage = []
peak_memory = []

for result in results:
    if result.dispatch_result:
        dr = result.dispatch_result
        duration = (dr.end_time - dr.start_time).total_seconds()
        durations.append(duration)
        memory_usage.append(dr.memory_usage)
        peak_memory.append(dr.peak_memory)

# Calculate statistics
print("Performance Statistics:")
print(f"Average Duration: {statistics.mean(durations):.2f}s")
print(f"Median Duration: {statistics.median(durations):.2f}s")
print(f"Average Memory: {statistics.mean(memory_usage):.1f} MB")
print(f"Peak Memory: {max(peak_memory):.1f} MB")
```

### When Available

⚠️ **Important:** `dispatch_result` is only populated when using:
- `arun_many()` method
- Custom dispatchers (MemoryAdaptiveDispatcher, SemaphoreDispatcher)

For single `arun()` calls, `dispatch_result` will be `None`.

---

## 📋 5. CrawlResult - Built-in Telemetry

**Documentation:** [CrawlResult Reference](https://docs.crawl4ai.com/api/crawl-result/)

### Overview

Every crawl operation returns a `CrawlResult` object with built-in telemetry data.

### Basic Metrics

```python
result = await crawler.arun(url="https://example.com")

# Status information
result.success              # bool: True if crawl succeeded
result.error_message        # str: Error description if failed
result.status_code          # int: HTTP status code (200, 404, etc.)
result.url                  # str: Final URL after redirects

# Response data
result.response_headers     # dict: HTTP response headers
result.html                 # str: Raw HTML content
result.cleaned_html         # str: Sanitized HTML
result.markdown             # MarkdownGenerationResult: Markdown content

# Media and links
result.media               # Dict[str, List[Dict]]: Images, videos, audio
result.links               # Dict[str, List[Dict]]: Internal/external links

# Session info
result.session_id          # str: Browser session identifier

# Optional captures
result.screenshot          # str: Base64 screenshot (if enabled)
result.pdf                 # bytes: PDF export (if enabled)
result.ssl_certificate     # SSLCertificate: SSL cert info (if enabled)
```

### Monitoring Example

```python
from datetime import datetime
import json

async def crawl_with_monitoring(url: str):
    """Crawl URL and log comprehensive metrics."""
    
    start_time = datetime.now()
    
    config = CrawlerRunConfig(
        capture_network_requests=True,
        capture_console_messages=True,
        screenshot=True
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create monitoring record
    metrics = {
        "timestamp": start_time.isoformat(),
        "url": result.url,
        "duration_seconds": duration,
        "success": result.success,
        "status_code": result.status_code,
        "error": result.error_message,
        
        # Content metrics
        "html_size_bytes": len(result.html) if result.html else 0,
        "cleaned_html_size": len(result.cleaned_html) if result.cleaned_html else 0,
        
        # Network metrics
        "network_requests": len(result.network_requests) if result.network_requests else 0,
        "console_messages": len(result.console_messages) if result.console_messages else 0,
        
        # Media counts
        "images_found": len(result.media.get("images", [])),
        "videos_found": len(result.media.get("videos", [])),
        
        # Links counts
        "internal_links": len(result.links.get("internal", [])),
        "external_links": len(result.links.get("external", [])),
        
        # Session info
        "session_id": result.session_id,
    }
    
    # Log to file
    with open("crawl_metrics.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")
    
    return result, metrics
```

---

## 🔧 6. Verbose Logging

### Overview

Enable detailed logging for debugging and monitoring.

### Configuration

```python
from crawl4ai import BrowserConfig, CrawlerRunConfig

# Browser-level logging
browser_config = BrowserConfig(
    verbose=True,  # Log browser operations
    # Additional debug output for browser lifecycle
)

# Crawl-level logging
crawler_config = CrawlerRunConfig(
    verbose=True,  # Log crawl operations
    # Additional debug output for extraction, caching, etc.
)

async with AsyncWebCrawler(config=browser_config) as crawler:
    result = await crawler.arun(url="...", config=crawler_config)
```

### Log Output Examples

```
[Browser] Launching Chromium in headless mode
[Browser] Viewport set to 1080x600
[Crawler] Navigating to https://example.com
[Crawler] Waiting for page load...
[Crawler] Page loaded in 1.23s
[Extraction] Starting content extraction
[Extraction] Found 42 images, 15 videos
[Cache] Writing result to cache: abc123.json
[Crawler] Crawl completed successfully
```

### Custom Logging Integration

```python
import logging
from crawl4ai import AsyncWebCrawler

# Configure Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("crawl_monitor")

async def monitored_crawl(url: str):
    """Crawl with custom logging."""
    logger.info(f"Starting crawl: {url}")
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            if result.success:
                logger.info(f"Crawl successful: {url}")
                logger.info(f"  Status: {result.status_code}")
                logger.info(f"  Content size: {len(result.html)} bytes")
            else:
                logger.error(f"Crawl failed: {url}")
                logger.error(f"  Error: {result.error_message}")
                
    except Exception as e:
        logger.exception(f"Exception during crawl: {url}")
        raise
```

---

## 🎯 7. Monitoring Integration Patterns

### Pattern 1: Time-Series Metrics Collection

```python
import time
from dataclasses import dataclass, asdict
from typing import List
import json

@dataclass
class CrawlMetric:
    timestamp: float
    url: str
    duration: float
    success: bool
    status_code: int
    html_size: int
    network_requests: int
    console_errors: int
    memory_mb: float = None

class MetricsCollector:
    """Collect and export crawler metrics."""
    
    def __init__(self):
        self.metrics: List[CrawlMetric] = []
    
    def record_crawl(self, url: str, result, duration: float):
        """Record metrics from a crawl result."""
        
        console_errors = 0
        if result.console_messages:
            console_errors = len([
                m for m in result.console_messages 
                if m.get("type") == "error"
            ])
        
        memory_mb = None
        if result.dispatch_result:
            memory_mb = result.dispatch_result.memory_usage
        
        metric = CrawlMetric(
            timestamp=time.time(),
            url=url,
            duration=duration,
            success=result.success,
            status_code=result.status_code or 0,
            html_size=len(result.html) if result.html else 0,
            network_requests=len(result.network_requests) if result.network_requests else 0,
            console_errors=console_errors,
            memory_mb=memory_mb
        )
        
        self.metrics.append(metric)
    
    def export_json(self, filename: str):
        """Export metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
    
    def export_csv(self, filename: str):
        """Export metrics to CSV file."""
        import csv
        
        if not self.metrics:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.metrics[0]).keys())
            writer.writeheader()
            writer.writerows([asdict(m) for m in self.metrics])

# Usage
collector = MetricsCollector()

async def monitored_crawl(url: str):
    start = time.time()
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
    
    duration = time.time() - start
    collector.record_crawl(url, result, duration)
    
    return result

# After crawling
collector.export_json("metrics.json")
collector.export_csv("metrics.csv")
```

### Pattern 2: Real-Time Dashboard Data

```python
import asyncio
from collections import deque
from datetime import datetime

class CrawlerDashboard:
    """Real-time crawler monitoring dashboard data."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_crawls = deque(maxlen=window_size)
        self.stats = {
            "total_crawls": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "total_duration": 0.0,
            "total_bytes": 0,
        }
    
    def update(self, result, duration: float):
        """Update dashboard with new crawl result."""
        
        self.stats["total_crawls"] += 1
        self.stats["total_duration"] += duration
        
        if result.success:
            self.stats["successful_crawls"] += 1
            if result.html:
                self.stats["total_bytes"] += len(result.html)
        else:
            self.stats["failed_crawls"] += 1
        
        self.recent_crawls.append({
            "timestamp": datetime.now().isoformat(),
            "url": result.url,
            "success": result.success,
            "duration": duration,
            "status_code": result.status_code,
        })
    
    def get_stats(self):
        """Get current statistics."""
        success_rate = (
            self.stats["successful_crawls"] / self.stats["total_crawls"] * 100
            if self.stats["total_crawls"] > 0 else 0
        )
        
        avg_duration = (
            self.stats["total_duration"] / self.stats["total_crawls"]
            if self.stats["total_crawls"] > 0 else 0
        )
        
        return {
            "total_crawls": self.stats["total_crawls"],
            "success_rate": f"{success_rate:.1f}%",
            "avg_duration": f"{avg_duration:.2f}s",
            "total_data": f"{self.stats['total_bytes'] / 1024 / 1024:.2f} MB",
            "recent_crawls": list(self.recent_crawls)[-10:],  # Last 10
        }

# Usage
dashboard = CrawlerDashboard()

async def monitored_crawl_with_dashboard(url: str):
    start = time.time()
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
    
    duration = time.time() - start
    dashboard.update(result, duration)
    
    # Print stats
    print(json.dumps(dashboard.get_stats(), indent=2))
```

### Pattern 3: Error Tracking & Alerting

```python
from enum import Enum
from typing import Callable, Optional

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorTracker:
    """Track and alert on crawler errors."""
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.alert_callback = alert_callback or print
        self.error_counts = {}
        self.consecutive_failures = 0
        self.failure_threshold = 5
    
    def track_result(self, result):
        """Track result and trigger alerts if needed."""
        
        if result.success:
            self.consecutive_failures = 0
            return
        
        # Failure detected
        self.consecutive_failures += 1
        
        # Track error type
        error_type = self._classify_error(result)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Alert on consecutive failures
        if self.consecutive_failures >= self.failure_threshold:
            self._alert(
                AlertLevel.CRITICAL,
                f"{self.consecutive_failures} consecutive failures detected"
            )
        
        # Alert on specific error types
        if result.status_code == 429:
            self._alert(AlertLevel.WARNING, f"Rate limited: {result.url}")
        elif result.status_code and result.status_code >= 500:
            self._alert(AlertLevel.ERROR, f"Server error {result.status_code}: {result.url}")
        
        # Check console errors
        if result.console_messages:
            errors = [m for m in result.console_messages if m.get("type") == "error"]
            if len(errors) > 10:
                self._alert(AlertLevel.WARNING, f"{len(errors)} JavaScript errors on {result.url}")
    
    def _classify_error(self, result):
        """Classify error type."""
        if result.status_code:
            if result.status_code == 404:
                return "not_found"
            elif result.status_code == 429:
                return "rate_limited"
            elif result.status_code >= 500:
                return "server_error"
            elif result.status_code >= 400:
                return "client_error"
        
        if result.error_message:
            if "timeout" in result.error_message.lower():
                return "timeout"
            elif "network" in result.error_message.lower():
                return "network_error"
        
        return "unknown_error"
    
    def _alert(self, level: AlertLevel, message: str):
        """Trigger alert."""
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "message": message,
            "error_counts": self.error_counts,
        }
        self.alert_callback(json.dumps(alert_data, indent=2))
    
    def get_summary(self):
        """Get error summary."""
        return {
            "consecutive_failures": self.consecutive_failures,
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
        }

# Usage
def my_alert_handler(alert_json: str):
    # Send to logging, webhook, Slack, etc.
    print(f"🚨 ALERT: {alert_json}")

tracker = ErrorTracker(alert_callback=my_alert_handler)

async def monitored_crawl_with_alerts(url: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        tracker.track_result(result)
        return result
```

---

## 📈 8. Integration with Existing Monitoring Systems

### Prometheus Integration (Custom)

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
crawl_requests = Counter('crawl_requests_total', 'Total crawl requests', ['status'])
crawl_duration = Histogram('crawl_duration_seconds', 'Crawl duration')
crawl_size = Histogram('crawl_content_bytes', 'Content size in bytes')
active_crawls = Gauge('crawl_active', 'Currently active crawls')
console_errors = Counter('crawl_console_errors_total', 'JavaScript errors detected')
network_failures = Counter('crawl_network_failures_total', 'Failed network requests')

async def monitored_crawl_prometheus(url: str):
    """Crawl with Prometheus metrics."""
    
    active_crawls.inc()
    start = time.time()
    
    try:
        config = CrawlerRunConfig(
            capture_network_requests=True,
            capture_console_messages=True
        )
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
        
        duration = time.time() - start
        
        # Record metrics
        status = "success" if result.success else "failure"
        crawl_requests.labels(status=status).inc()
        crawl_duration.observe(duration)
        
        if result.html:
            crawl_size.observe(len(result.html))
        
        # Track errors
        if result.console_messages:
            error_count = len([m for m in result.console_messages if m.get("type") == "error"])
            console_errors.inc(error_count)
        
        if result.network_requests:
            failure_count = len([r for r in result.network_requests if r.get("event_type") == "request_failed"])
            network_failures.inc(failure_count)
        
        return result
        
    finally:
        active_crawls.dec()

# Start Prometheus HTTP server
start_http_server(8000)
```

### OpenTelemetry Integration (Custom)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add span processor
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)

async def monitored_crawl_otel(url: str):
    """Crawl with OpenTelemetry tracing."""
    
    with tracer.start_as_current_span("crawl_url") as span:
        span.set_attribute("url", url)
        
        start = time.time()
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        
        duration = time.time() - start
        
        # Add span attributes
        span.set_attribute("success", result.success)
        span.set_attribute("status_code", result.status_code or 0)
        span.set_attribute("duration_ms", duration * 1000)
        span.set_attribute("content_size", len(result.html) if result.html else 0)
        
        if not result.success:
            span.set_status(trace.Status(trace.StatusCode.ERROR, result.error_message))
        
        return result
```

---

## 🔍 9. Limitations & Missing Features

### What's NOT Available in Crawl4AI

❌ **Built-in Prometheus Exporter**
- No native `/metrics` endpoint
- Must implement custom Prometheus client integration

❌ **OpenTelemetry Native Support**
- No built-in tracing/spans
- Must wrap with custom OpenTelemetry instrumentation

❌ **Web-based Dashboard**
- No UI for monitoring
- Must build custom dashboard or use external tools

❌ **Alert/Webhook System**
- No native alerting mechanism
- Must implement custom alert handlers

❌ **Distributed Tracing**
- No trace context propagation
- Limited to single-process monitoring

❌ **APM Integration**
- No native Datadog, New Relic, or Dynatrace integration
- Must use custom instrumentation

❌ **Log Aggregation**
- No native ELK or Loki integration
- Uses standard Python logging (can be configured externally)

---

## ✅ 10. Recommendations for Implementation

### For Development/Debugging

1. **Enable all capture features:**
```python
CrawlerRunConfig(
    capture_network_requests=True,
    capture_console_messages=True,
    screenshot=True,
    verbose=True
)
```

2. **Use CrawlerMonitor for visibility**
3. **Export monitoring data to JSON for analysis**

### For Production

1. **Implement custom Prometheus integration** (see section 8)
2. **Use batch crawling with resource limits:**
```python
CrawlerRunConfig(
    memory_threshold_percent=70.0,
    max_session_permit=20,
    enable_rate_limiting=True
)
```

3. **Set up error tracking and alerting** (see Pattern 3)
4. **Monitor dispatch_result metrics** for performance insights
5. **Implement custom logging to centralized system**

### Monitoring Architecture Suggestion

```
Crawl4AI Application
       ↓
  Custom Metrics Layer
       ↓
   ┌───┴───┐
   ↓       ↓
Prometheus  Logs
   ↓       ↓
Grafana   ELK/Loki
```

---

## 📚 11. References

### Official Documentation

- [Crawl4AI Documentation](https://docs.crawl4ai.com/)
- [Network & Console Capture](https://docs.crawl4ai.com/advanced/network-console-capture/)
- [Multi-URL Crawling](https://docs.crawl4ai.com/advanced/multi-url-crawling/)
- [CrawlResult Reference](https://docs.crawl4ai.com/api/crawl-result/)
- [AsyncWebCrawler API](https://docs.crawl4ai.com/api/async-webcrawler/)
- [Configuration Parameters](https://docs.crawl4ai.com/api/parameters/)

### Code Examples

- [Crawler Monitor Example](https://github.com/unclecode/crawl4ai/blob/main/docs/examples/crawler_monitor_example.py)
- [Network Console Capture Example](https://github.com/unclecode/crawl4ai/blob/main/docs/examples/network_console_capture_example.py)
- [All Examples Directory](https://github.com/unclecode/crawl4ai/tree/main/docs/examples/)

### Related Components

- `crawl4ai.components.crawler_monitor.CrawlerMonitor`
- `crawl4ai.models.CrawlStatus`
- `crawl4ai.models.CrawlResult`
- `crawl4ai.models.DispatchResult`

---

## 📝 Appendix: Quick Reference

### Enable Full Monitoring

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# Browser config
browser_config = BrowserConfig(
    verbose=True,
    headless=True
)

# Crawler config with all monitoring enabled
crawler_config = CrawlerRunConfig(
    # Capture
    capture_network_requests=True,
    capture_console_messages=True,
    screenshot=True,
    
    # Resource monitoring
    memory_threshold_percent=70.0,
    check_interval=1.0,
    max_session_permit=20,
    
    # Rate limiting
    enable_rate_limiting=True,
    
    # Verbose output
    verbose=True
)

async with AsyncWebCrawler(config=browser_config) as crawler:
    result = await crawler.arun(url="https://example.com", config=crawler_config)
    
    # Access all monitoring data
    print(f"Success: {result.success}")
    print(f"Status: {result.status_code}")
    print(f"Network Events: {len(result.network_requests or [])}")
    print(f"Console Messages: {len(result.console_messages or [])}")
    
    if result.dispatch_result:
        dr = result.dispatch_result
        print(f"Memory: {dr.memory_usage:.1f} MB (Peak: {dr.peak_memory:.1f} MB)")
        print(f"Duration: {(dr.end_time - dr.start_time).total_seconds():.2f}s")
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-28  
**Crawl4AI Version:** 0.7.x
