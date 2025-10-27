# Working with Crawl4AI Repository

## Repository Structure

```
crawl4ai-rag-mcp/
├── src/
│   ├── services/     # Crawling, search logic
│   ├── database/     # Vector DB adapters
│   ├── utils/        # Helper functions
│   └── tools.py      # MCP tool definitions
└── docs/             # Documentation
```

## Deployment Workflow

```bash
# Make changes
vim src/services/crawling.py

# Commit
git add .
git commit -m "fix: your message"

# Push (auto-deploys to Coolify)
git push origin main
```

Coolify watches `main` branch and deploys automatically.

## Testing

```bash
# Run tests
python -m pytest tests/

# Check deployment
curl https://rag.melo.eu.org/health
```

## Notes

- Deploy time: ~2-5 minutes
- Monitor: Coolify dashboard
- Server: https://rag.melo.eu.org
