#\!/bin/bash
docker run --rm --network host \
  -e NEO4J_URI=bolt://localhost:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=your_password \
  -e USE_KNOWLEDGE_GRAPH=true \
  -e OPENAI_API_KEY=sk-dummy-key-for-testing \
  -e QDRANT_HOST=localhost \
  -e QDRANT_PORT=6333 \
  mcp-crawl4ai:test 2>&1 | head -100
