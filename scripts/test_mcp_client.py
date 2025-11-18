#!/usr/bin/env python3
"""
Simple MCP client for testing the Crawl4AI MCP server.

Uses FastMCP client to connect to the server and test tools.
"""

import asyncio
import json

import httpx


async def test_http_mcp():
    """Test MCP server over HTTP using direct HTTP requests."""
    url = "http://localhost:8051/mcp"

    # Create headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    async with httpx.AsyncClient() as client:
        # Step 1: Initialize session
        print("1. Initializing MCP session...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0",
                },
            },
        }

        response = await client.post(url, headers=headers, json=init_request, timeout=30.0)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        # Extract session ID
        session_id = response.headers.get("mcp-session-id")
        if not session_id:
            print("ERROR: No session ID in response")
            print(response.text)
            return

        print(f"✓ Session ID: {session_id}\n")

        # Add session ID to headers
        headers["mcp-session-id"] = session_id

        # Parse SSE response
        lines = response.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:"):
                data = json.loads(line[6:])  # Skip "data: "
                print(f"Initialize response: {json.dumps(data, indent=2)}\n")

        # Step 2: Send initialized notification
        print("2. Sending initialized notification...")
        notif_request = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }

        response = await client.post(url, headers=headers, json=notif_request, timeout=30.0)
        print(f"Status: {response.status_code}\n")

        # Step 3: List tools
        print("3. Listing available tools...")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = await client.post(url, headers=headers, json=tools_request, timeout=30.0)
        print(f"Status: {response.status_code}")

        # Parse SSE response
        lines = response.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:"):
                data = json.loads(line[6:])
                if "result" in data:
                    tools = data["result"].get("tools", [])
                    print(f"✓ Found {len(tools)} tools:")
                    for tool in tools[:5]:  # Show first 5
                        print(f"  - {tool['name']}: {tool.get('description', 'No description')[:80]}")
                    if len(tools) > 5:
                        print(f"  ... and {len(tools) - 5} more")
                elif "error" in data:
                    print(f"ERROR: {data['error']}")
                    return

        print("\n4. Testing agentic_search tool (if available)...")
        # Try to call agentic_search
        search_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "agentic_search",
                "arguments": {
                    "query": "What is LLM reasoning?",
                    "max_iterations": 1,
                    "completeness_threshold": 0.8,
                },
            },
        }

        response = await client.post(url, headers=headers, json=search_request, timeout=120.0)
        print(f"Status: {response.status_code}")

        # Parse SSE response
        lines = response.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:"):
                data = json.loads(line[6:])
                if "result" in data:
                    # Debug: print full result structure
                    print(f"DEBUG - Full result: {json.dumps(data['result'], indent=2)}")

                    # Check if result is a list or dict
                    result = data["result"]

                    # Handle different MCP response formats
                    if isinstance(result, dict):
                        # Format: {"content": [{"type": "text", "text": "..."}]}
                        if "content" in result:
                            content_list = result["content"]
                            if isinstance(content_list, list) and len(content_list) > 0:
                                result_content = content_list[0].get("text", "")
                            else:
                                result_content = result.get("structuredContent", {}).get("result", "{}")
                        else:
                            result_content = result.get("structuredContent", {}).get("result", str(result))
                    elif isinstance(result, list) and len(result) > 0:
                        result_content = result[0].get("content", "")
                    else:
                        print(f"WARN: Unexpected result structure: {result}")
                        continue

                    # Parse the content if it's JSON string
                    if isinstance(result_content, str):
                        try:
                            result_data = json.loads(result_content)
                        except json.JSONDecodeError:
                            result_data = {"raw": result_content}
                    else:
                        result_data = result_content

                    print("✓ Agentic search result:")
                    print(f"  Success: {result_data.get('success')}")
                    print(f"  Status: {result_data.get('status')}")
                    print(f"  Completeness: {result_data.get('completeness', 0):.2f}")
                    if not result_data.get("success"):
                        print(f"  Error: {result_data.get('error')}")
                elif "error" in data:
                    print(f"ERROR: {data['error']}")


if __name__ == "__main__":
    asyncio.run(test_http_mcp())
