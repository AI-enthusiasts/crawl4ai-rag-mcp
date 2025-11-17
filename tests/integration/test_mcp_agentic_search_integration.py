"""Integration test for MCP agentic_search tool.

This test reproduces the bug where agentic_search fails with
"Completeness evaluation failed" error.

NO MOCKS - Real MCP server integration test.

Requirements:
- MCP server running on localhost:8051
- Qdrant running on localhost:6333
- OPENAI_API_KEY set in environment
"""

import asyncio
import json
import os

import httpx
import pytest


@pytest.fixture
async def mcp_session():
    """Create MCP session with the server."""
    url = "http://localhost:8051/mcp"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check if MCP server is available
        try:
            health_check = await client.get("http://localhost:8051/health", timeout=5.0)
            if health_check.status_code != 200:
                pytest.skip("MCP server not available at localhost:8051")
        except Exception as e:
            pytest.skip(f"MCP server not available: {e}")

        # Initialize session
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = await client.post(url, headers=headers, json=init_request)
        assert response.status_code == 200, f"Initialize failed: {response.text}"

        # Extract session ID
        session_id = response.headers.get("mcp-session-id")
        assert session_id, "No session ID in response"

        headers["mcp-session-id"] = session_id

        # Send initialized notification
        notif_request = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }

        await client.post(url, headers=headers, json=notif_request)

        yield (client, url, headers)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_agentic_search_completeness_evaluation(mcp_session):
    """Test that agentic_search completes without 'Completeness evaluation failed' error.

    This test will FAIL if the bug exists.
    When working correctly, should return success=true with results or proper error handling.

    Bug symptoms:
    - success: false
    - status: "error"
    - error: "Completeness evaluation failed"
    - completeness: 0.0
    """
    client, url, headers = mcp_session

    # Call agentic_search tool
    search_request = {
        "jsonrpc": "2.0",
        "id": 2,
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

    response = await client.post(url, headers=headers, json=search_request)
    assert response.status_code == 200, f"Tool call failed: {response.text}"

    # Parse SSE response
    lines = response.text.strip().split("\n")
    result_data = None

    for line in lines:
        if line.startswith("data:"):
            data = json.loads(line[6:])  # Skip "data: "

            if "error" in data:
                pytest.fail(f"MCP error: {data['error']}")

            if "result" in data:
                result = data["result"]

                # Extract content from MCP response
                if isinstance(result, dict) and "content" in result:
                    content_list = result["content"]
                    if isinstance(content_list, list) and len(content_list) > 0:
                        result_text = content_list[0].get("text", "")
                        result_data = json.loads(result_text)

    assert result_data is not None, "No result data in response"

    # Check for the bug
    if not result_data.get("success"):
        error = result_data.get("error", "")
        status = result_data.get("status", "")

        # BUG REPRODUCED: This assertion will FAIL if bug exists
        assert (
            error != "Completeness evaluation failed"
        ), f"BUG REPRODUCED: {error} (status={status}, completeness={result_data.get('completeness', 0)})"

        # If not the completeness bug, check for expected errors
        assert error, "Failed without error message"
        # Other errors are acceptable (e.g., network timeout, rate limit, etc.)
        print(f"Test passed with expected error: {error}")
    else:
        # Success case
        assert result_data["success"] is True
        assert result_data.get("completeness", 0) > 0
        print(f"Test passed: completeness={result_data.get('completeness')}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mcp_server_connectivity():
    """Test basic MCP server connectivity.

    This test verifies the MCP server is accessible and responding.
    """
    url = "http://localhost:8051/mcp"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        try:
            response = await client.post(url, headers=headers, json=init_request)
            assert response.status_code == 200
            assert "mcp-session-id" in response.headers
        except httpx.ConnectError:
            pytest.skip("MCP server not running on localhost:8051")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
