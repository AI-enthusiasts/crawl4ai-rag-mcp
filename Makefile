# Crawl4AI MCP Server - Makefile with 2025 Best Practices
SHELL := /bin/bash
.DELETE_ON_ERROR:  # Clean up on failure  
.ONESHELL:         # Run recipes in single shell

# ============================================
# Variables
# ============================================
APP_NAME := crawl4ai-mcp
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "0.1.0")
REGISTRY := docker.io/krashnicov
IMAGE := $(REGISTRY)/$(APP_NAME)
PLATFORMS := linux/amd64,linux/arm64

# Docker compose command
DOCKER_COMPOSE := docker compose
PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff

# ============================================
# PHONY Targets (Best Practice)
# ============================================
.PHONY: help install start stop clean test build push release
.PHONY: dev prod logs health security-scan
.PHONY: docker-build docker-push docker-scan build-local build-prod
.PHONY: dirs env-setup quickstart update
.PHONY: restart status shell python lint format
.PHONY: dev-bg dev-logs dev-down dev-restart dev-rebuild
.PHONY: test-unit test-integration test-all test-coverage
.PHONY: clean-all env-check deps ps
.PHONY: prod-down prod-logs prod-ps prod-restart
.PHONY: start-full start-dev volumes backup restore
.PHONY: dev-services dev-stdio dev-hybrid dev-services-down dev-services-logs dev-setup-stdio
.PHONY: help-legacy test-quick

# ============================================
# Default Target
# ============================================
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "╔════════════════════════════════════════════╗"
	@echo "║   Crawl4AI MCP Server - Make Commands     ║"
	@echo "╚════════════════════════════════════════════╝"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install        # First-time setup"
	@echo "  make start          # Start services"
	@echo "  make logs           # View logs"
	@echo "  make stop           # Stop services"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "For detailed help on legacy commands, run: make help-legacy"

help-legacy: ## Show legacy command help
	@echo "Legacy Development Commands"
	@echo "============================================"
	@echo ""
	@echo "Development Environment:"
	@echo "  make dev             - Start development with watch mode"
	@echo "  make dev-bg          - Start development in background"
	@echo "  make dev-logs        - View development logs"
	@echo "  make dev-down        - Stop development environment"
	@echo "  make dev-restart     - Restart development services"
	@echo "  make dev-rebuild     - Rebuild development environment"
	@echo ""
	@echo "Testing:"
	@echo "  make test-unit       - Run unit tests"
	@echo "  make test-integration- Run integration tests"
	@echo "  make test-all        - Run all tests"
	@echo "  make test-coverage   - Run tests with coverage"

# ============================================
# Installation & Setup (NEW)
# ============================================
dirs: ## Create required directories
	@echo "Creating directory structure..."
	@mkdir -p data
	@mkdir -p logs
	@mkdir -p analysis_scripts/{user_scripts,validation_results}
	@mkdir -p docker/neo4j/import
	@mkdir -p notebooks
	@echo "✓ Directories created"

env-setup: ## Setup environment file
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		cp .env.example .env; \
		echo "✓ Environment file created"; \
		echo "⚠ Please edit .env with your API keys"; \
	else \
		echo "✓ Environment file exists"; \
	fi

env-check: ## Validate environment variables
	@echo "Checking environment configuration..."
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found"; \
		echo "Creating from template..."; \
		cp .env.example .env; \
		echo "✓ Created .env file - please configure your API keys"; \
		exit 1; \
	fi
	@echo "✓ Environment configured"

install: dirs env-setup ## One-click installation
	@echo "╔════════════════════════════════════════════╗"
	@echo "║     Installing Crawl4AI MCP Server        ║"
	@echo "╚════════════════════════════════════════════╝"
	@echo ""
	@echo "Checking Docker..."
	@docker --version || (echo "✗ Docker not installed" && exit 1)
	@docker compose version || (echo "✗ Docker Compose not installed" && exit 1)
	@echo "✓ Docker ready"
	@echo ""
	@echo "Pulling images..."
	@docker compose pull
	@echo ""
	@echo "✅ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env file with your API keys"
	@echo "  2. Run 'make start' to start services"
	@echo ""

quickstart: install start ## Complete setup and start

update: ## Pull latest code and rebuild production image
	@echo "╔════════════════════════════════════════════╗"
	@echo "║   Updating Crawl4AI MCP Server            ║"
	@echo "╚════════════════════════════════════════════╝"
	@echo ""
	@echo "Pulling latest code from git..."
	@git pull
	@echo ""
	@echo "Building new production image..."
	@$(MAKE) build-prod
	@echo ""
	@echo "✅ Update complete!"
	@echo ""
	@echo "Run 'make restart' to apply changes"

# ============================================
# Service Management (NEW SIMPLIFIED)
# ============================================
start: ## Start core services (includes Neo4j)
	@echo "Starting services..."
	@docker compose --profile core up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@$(MAKE) health
	@echo ""
	@echo "🚀 Services running at:"
	@echo "  • MCP Server: http://localhost:8051"
	@echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "  • Neo4j Browser: http://localhost:7474"
	@echo "  • SearXNG Search: http://localhost:8080"
	@echo ""

start-full: ## Start full services (same as core now)
	@$(MAKE) start

start-dev: ## Start development environment with all tools
	@echo "Starting development environment..."
	@docker compose --profile dev up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@$(MAKE) health
	@echo ""
	@echo "🚀 Development services running at:"
	@echo "  • MCP Server: http://localhost:8051"
	@echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "  • Neo4j Browser: http://localhost:7474"
	@echo "  • SearXNG Search: http://localhost:8080"
	@echo "  • Mailhog UI: http://localhost:8025"
	@echo "  • Jupyter Lab: http://localhost:8888 (token: crawl4ai)"
	@echo ""

stop: ## Stop all services
	@echo "Stopping services..."
	@docker compose --profile core --profile dev down
	@echo "✓ Services stopped"

restart: stop start ## Restart services

logs: ## View service logs
	@docker compose logs -f --tail=100

health: ## Check service health
	@echo "Checking service health..."
	@docker compose ps --format "table {{.Name}}\t{{.Status}}"

status: health ## Alias for health

ps: status ## Show running containers

# ============================================
# Volume Management (NEW)
# ============================================
volumes: ## List all Docker volumes for this project
	@echo "Project volumes:"
	@docker volume ls | grep -E "crawl4ai" || echo "No volumes found"

backup: ## Backup data volumes to ./backups directory
	@echo "Creating backup..."
	@mkdir -p backups/$(shell date +%Y%m%d-%H%M%S)
	@cd backups/$(shell date +%Y%m%d-%H%M%S) && \
		docker run --rm -v crawl4ai_mcp_qdrant-data:/data -v $$(pwd):/backup alpine tar czf /backup/qdrant-data.tar.gz -C /data . && \
		docker run --rm -v crawl4ai_mcp_valkey-data:/data -v $$(pwd):/backup alpine tar czf /backup/valkey-data.tar.gz -C /data . && \
		docker run --rm -v crawl4ai_mcp_neo4j-data:/data -v $$(pwd):/backup alpine tar czf /backup/neo4j-data.tar.gz -C /data .
	@echo "✓ Backup complete in backups/"

restore: ## Restore data volumes from backup (specify BACKUP_DIR)
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo "Error: Specify BACKUP_DIR=backups/YYYYMMDD-HHMMSS"; \
		exit 1; \
	fi
	@echo "Restoring from $(BACKUP_DIR)..."
	@docker compose --profile core --profile dev down
	@docker run --rm -v crawl4ai_mcp_qdrant-data:/data -v $$(pwd)/$(BACKUP_DIR):/backup alpine tar xzf /backup/qdrant-data.tar.gz -C /data
	@docker run --rm -v crawl4ai_mcp_valkey-data:/data -v $$(pwd)/$(BACKUP_DIR):/backup alpine tar xzf /backup/valkey-data.tar.gz -C /data
	@docker run --rm -v crawl4ai_mcp_neo4j-data:/data -v $$(pwd)/$(BACKUP_DIR):/backup alpine tar xzf /backup/neo4j-data.tar.gz -C /data
	@echo "✓ Restore complete"

# ============================================
# Development Environment (UPDATED)
# ============================================
dev: start-dev ## Start development environment

dev-bg: ## Start development in background with watch
	@echo "Starting development environment in background..."
	@docker compose --profile dev up -d --build
	@echo "Starting watch mode..."
	@docker compose --profile dev watch

dev-logs: ## View development logs
	@docker compose logs -f mcp-crawl4ai

dev-down: ## Stop development environment
	@echo "Stopping development environment..."
	@docker compose --profile dev down

dev-restart: ## Restart development services
	@echo "Restarting development services..."
	@docker compose restart mcp-crawl4ai

dev-rebuild: ## Rebuild development environment
	@echo "Rebuilding development environment..."
	@docker compose --profile dev down
	@docker compose build --no-cache mcp-crawl4ai
	@$(MAKE) dev

# ============================================
# Hybrid Development (stdio mode)
# ============================================
dev-services: ## Start only database services (for stdio development)
	@echo "Starting database services for stdio development..."
	@docker compose --profile services-only up -d
	@echo "Services started. Waiting for health checks..."
	@sleep 5
	@echo "Services available at:"
	@echo "  - Qdrant:   http://localhost:6333/dashboard"
	@echo "  - Neo4j:    http://localhost:7474"
	@echo "  - SearXNG:  http://localhost:8080"
	@echo "  - Valkey:   localhost:6379"
	@echo ""
	@echo "Run 'make dev-stdio' to start MCP server in stdio mode"

dev-stdio: ## Run MCP server locally with stdio transport
	@echo "Starting MCP server in stdio mode..."
	@echo "Using configuration from .env.dev"
	@if [ ! -f .env.dev ]; then \
		echo "Error: .env.dev not found. Run 'make dev-setup-stdio' first."; \
		exit 1; \
	fi
	@export $$(cat .env.dev | grep -v '^\#' | xargs) && uv run python src/main.py

dev-hybrid: dev-services ## Start services and run MCP in stdio mode
	@echo "Starting hybrid development environment..."
	@$(MAKE) dev-stdio

dev-services-down: ## Stop database services
	@echo "Stopping database services..."
	@docker compose --profile services-only down

dev-services-logs: ## View logs for database services
	@docker compose --profile services-only logs -f

dev-setup-stdio: ## Initial setup for stdio development
	@echo "Setting up stdio development environment..."
	@if [ ! -f .env.dev ]; then \
		echo "Creating .env.dev from template..."; \
		cp .env.example .env.dev; \
		sed -i 's/TRANSPORT=.*/TRANSPORT=stdio/' .env.dev; \
		sed -i 's|SEARXNG_URL=.*|SEARXNG_URL=http://localhost:8080|' .env.dev; \
		sed -i 's|QDRANT_URL=.*|QDRANT_URL=http://localhost:6333|' .env.dev; \
		sed -i 's|NEO4J_URI=.*|NEO4J_URI=bolt://localhost:7687|' .env.dev; \
		echo ".env.dev created. Please update OPENAI_API_KEY if needed."; \
	else \
		echo ".env.dev already exists."; \
	fi

# ============================================
# Production Environment (UPDATED)
# ============================================
prod: start ## Start production environment (alias for start)

prod-down: stop ## Stop production environment (alias for stop)

prod-logs: logs ## View production logs (alias for logs)

prod-ps: ps ## Show production containers (alias for ps)

prod-restart: restart ## Restart production services (alias for restart)

# ============================================
# Testing
# ============================================
test: test-unit ## Run unit tests (alias)

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	@if [ -f /.dockerenv ]; then \
		$(PYTEST) tests/unit -v --tb=short; \
	else \
		$(DOCKER_COMPOSE) run --rm mcp-crawl4ai $(PYTEST) tests/unit -v --tb=short; \
	fi

test-integration: ## Run integration tests
	@echo "Running integration tests with Docker services..."
	$(DOCKER_COMPOSE) --profile core up -d
	$(DOCKER_COMPOSE) run --rm mcp-crawl4ai $(PYTEST) tests/integration -v
	$(DOCKER_COMPOSE) --profile core down

test-all: ## Run all tests
	@echo "Running all tests..."
	$(MAKE) test-unit
	$(MAKE) test-integration

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	@docker compose run --rm mcp-crawl4ai uv run pytest --cov=src --cov-fail-under=80

test-quick: ## Run quick unit tests
	@docker compose run --rm mcp-crawl4ai uv run pytest tests/unit -v

# ============================================
# Docker Build & Release (NEW)
# ============================================
build-local: ## Build Docker image locally
	@echo "Building local Docker image..."
	@docker compose build mcp-crawl4ai
	@echo "✓ Local build complete"

build-prod: ## Build production image (use after pulling repo updates)
	@echo "╔════════════════════════════════════════════╗"
	@echo "║   Building Production Docker Image        ║"
	@echo "╚════════════════════════════════════════════╝"
	@echo ""
	@echo "Pulling latest base images..."
	@docker compose pull
	@echo ""
	@echo "Building production image with no cache..."
	@docker compose build --no-cache mcp-crawl4ai
	@echo ""
	@echo "✅ Build complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make stop' to stop existing services"
	@echo "  2. Run 'make start' to start with new image"
	@echo "  3. Run 'make health' to verify services"

docker-build: ## Build Docker image for multiple platforms
	@echo "Building multi-platform Docker image..."
	@docker buildx create --use --name multiarch || true
	@docker buildx build \
		--platform $(PLATFORMS) \
		--tag $(IMAGE):$(VERSION) \
		--tag $(IMAGE):latest \
		--cache-from type=registry,ref=$(IMAGE):buildcache \
		--cache-to type=registry,ref=$(IMAGE):buildcache,mode=max \
		--load .
	@echo "✓ Multi-platform build complete"

docker-push: ## Push to Docker Hub
	@echo "Pushing to Docker Hub..."
	@docker push $(IMAGE):$(VERSION)
	@docker push $(IMAGE):latest
	@echo "✓ Push complete"

docker-scan: ## Security scan with Trivy
	@echo "Running security scan..."
	@docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image --severity HIGH,CRITICAL $(IMAGE):$(VERSION)

build: docker-build ## Alias for docker-build

push: docker-push ## Alias for docker-push

security-scan: docker-scan ## Alias for docker-scan

release: test security-scan build push ## Complete release process
	@echo "✅ Release complete!"
	@echo "Next steps:"
	@echo "  1. Create GitHub release"
	@echo "  2. Update changelog"
	@echo "  3. Tag the commit"

# ============================================
# Development Helpers
# ============================================
shell: ## Open shell in container
	@docker compose exec mcp-crawl4ai /bin/bash || \
		docker compose run --rm mcp-crawl4ai /bin/bash

python: ## Open Python REPL
	@docker compose exec mcp-crawl4ai python || \
		docker compose run --rm mcp-crawl4ai python

lint: ## Run code linting
	@echo "Running linter..."
	@docker compose run --rm mcp-crawl4ai uv run ruff check src/ tests/

format: ## Format code
	@echo "Formatting code..."
	@docker compose run --rm mcp-crawl4ai uv run ruff format src/ tests/

deps: ## Install/update dependencies
	@echo "Installing dependencies..."
	@uv sync

# ============================================
# Cleanup
# ============================================
clean: ## Clean test artifacts and caches
	@echo "Cleaning test artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov coverage.xml .coverage.* 2>/dev/null || true
	@echo "✓ Cleanup complete"

clean-all: stop clean ## Clean everything including volumes
	@echo "⚠ WARNING: This will delete all data!"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose --profile core --profile dev down -v; \
		rm -rf data logs; \
		echo "✓ All data cleaned"; \
	else \
		echo "Cancelled"; \
	fi