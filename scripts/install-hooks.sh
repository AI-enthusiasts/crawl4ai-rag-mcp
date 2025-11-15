#!/bin/bash
# Install git hooks for Crawl4AI RAG MCP development

set -e

HOOK_DIR=".git/hooks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📝 Installing Git hooks..."

# Create pre-commit hook
cat > "$HOOK_DIR/pre-commit" << 'HOOK'
#!/bin/bash
# Pre-commit hook for Crawl4AI RAG MCP
# Runs import tests and basic linting before allowing commit

set -e

echo "🔍 Running pre-commit checks..."

# 1. Import verification (fast, critical)
echo "📦 Verifying module imports..."
if ! uv run pytest tests/test_imports.py --tb=short -q; then
    echo "❌ Import test failed! Fix import errors before committing."
    exit 1
fi
echo "✅ All modules import successfully"

# 2. Ruff linting (fast)
echo "🔧 Running ruff linter..."
if ! uv run ruff check src/ --quiet; then
    echo "⚠️  Ruff found issues. Run 'uv run ruff check src/ --fix' to auto-fix."
    echo "   Continuing anyway (warnings only)..."
fi

# 3. Quick unit tests (optional, can be disabled)
# Uncomment to run quick unit tests on every commit
# echo "🧪 Running quick unit tests..."
# if ! uv run pytest tests/test_imports.py tests/database/test_qdrant_adapter_comprehensive.py -q; then
#     echo "❌ Quick tests failed!"
#     exit 1
# fi

echo "✅ All pre-commit checks passed!"
exit 0
HOOK

chmod +x "$HOOK_DIR/pre-commit"

echo "✅ Git hooks installed successfully!"
echo ""
echo "To skip hooks for a specific commit, use: git commit --no-verify"
