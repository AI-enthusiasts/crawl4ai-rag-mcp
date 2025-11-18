#!/usr/bin/env python3
"""
HTTP-to-STDIO bridge for Claude Desktop MCP integration.
Connects to HTTP MCP server and translates to STDIO for Claude Desktop.
"""

import asyncio
import json
import sys
from typing import Any

import aiohttp

MCP_SERVER_URL = "http://localhost:8051/mcp"

async def send_request(session: aiohttp.ClientSession, request: dict[str, Any]) -> dict[str, Any]:
    """Send request to HTTP MCP server and return response."""
    async with session.post(MCP_SERVER_URL, json=request) as resp:
        return await resp.json()

async def main():
    """Main bridge loop - read from stdin, forward to HTTP, write to stdout."""
    async with aiohttp.ClientSession() as session:
        # Read from stdin line by line
        for line in sys.stdin:
            try:
                # Parse JSON-RPC request
                request = json.loads(line.strip())

                # Forward to HTTP server
                response = await send_request(session, request)

                # Write response to stdout
                print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                # Send error response
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e!s}",
                    },
                    "id": None,
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                # Send general error response
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {e!s}",
                    },
                    "id": request.get("id") if "request" in locals() else None,
                }
                print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
