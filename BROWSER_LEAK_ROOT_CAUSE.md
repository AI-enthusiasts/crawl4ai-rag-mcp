# Browser Process Leak - Root Cause Analysis

## Problem Summary
Browser processes accumulate over time, growing from 8 to 300+ processes, consuming 99% memory.

## Root Cause Identified

### Double Lifespan Initialization

**File**: `src/main.py`

**Line 34**: FastMCP created with lifespan
```python
mcp = FastMCP("Crawl4AI MCP Server", lifespan=crawl4ai_lifespan)
```

**Line 71**: Manual lifespan call (PROBLEM!)
```python
async with crawl4ai_lifespan(mcp) as context:
    # ... 310 lines of code ...
    await mcp.run_http_async(...)  # This ALSO calls lifespan!
```

### What Happens

1. **FastMCP stores lifespan** in constructor (line 34)
2. **Manual call** creates first crawler (line 71)
3. **Each new MCP HTTP session** triggers the manual block again
4. **run_http_async()** calls `_lifespan_manager()` which calls lifespan again
5. **Result**: Multiple crawler instances, none properly cleaned up

### Evidence

**Timeline of crawler initializations** (from logs):
```
14:53:17 - Crawler init #1 (app start)
14:56:06 - Crawler init #2 (new MCP session) +3 min
14:58:53 - Crawler init #3 (new MCP session) +3 min
15:05:56 - Crawler init #4 (new MCP session) +7 min
15:10:00 - Crawler init #5 (new MCP session) +4 min
15:12:27 - Crawler init #6 (new MCP session) +2 min
15:21:27 - Crawler init #7 (new MCP session) +9 min
```

**7 initializations in 38 minutes** = 7 browser instances running simultaneously!

**Resource growth**:
- Start: 8 processes, 603MB (14.7%)
- After 38min: 63 processes, 1.29GB (32.2%)
- Projected: 300+ processes, 4GB (99%) after few hours

## Solution

### Remove Manual Lifespan Call

Per [FastMCP documentation](https://gofastmcp.com/deployment/http):

> ❌ **WRONG**: Don't manually manage lifespan
> ```python
> async with mcp._lifespan(mcp):
>     await mcp.run_http_async(...)
> ```

> ✅ **CORRECT**: lifespan is handled automatically
> ```python
> mcp.run(transport="http", host="0.0.0.0", port=8000)
> ```

### Required Changes

**Remove lines 71-397** (the entire `async with crawl4ai_lifespan` block) and unindent the contents.

**Before**:
```python
if transport == "http":
    async with crawl4ai_lifespan(mcp) as context:
        middleware = []
        # ... 310 lines ...
        await mcp.run_http_async(...)
```

**After**:
```python
if transport == "http":
    middleware = []
    # ... 310 lines ...
    await mcp.run_http_async(...)
    # FastMCP automatically calls crawl4ai_lifespan ONCE at startup
```

## Why This Wasn't Caught Earlier

1. **Misleading comment** (line 68-69): "HTTP mode doesn't automatically call lifespan managers" - This is FALSE
2. **Works initially**: First crawler works fine, problem only appears over time
3. **Gradual accumulation**: Not immediately obvious, takes hours to manifest
4. **Complex interaction**: Requires understanding both FastMCP and crawl4ai lifecycles

## References

- FastMCP Lifespan Docs: https://gofastmcp.com/deployment/http
- Crawl4AI Issue #943: https://github.com/unclecode/crawl4ai/issues/943
- FastMCP Source: `_lifespan_manager()` in server.py

## Impact

**Before fix**: 
- 300+ browser processes
- 99% memory usage
- Server becomes unresponsive
- Requires restart every few hours

**After fix**:
- Single browser instance
- Stable 8-10 processes
- ~600MB memory (15%)
- Can run indefinitely

## Status

- ✅ Root cause identified
- ✅ Solution documented
- ⏳ Implementation pending (requires refactoring 310 lines)
- ⏳ Testing required after implementation
