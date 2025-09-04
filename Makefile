.PHONY: help install setup test run-mcp run-api run-both clean docker-build docker-run docker-stop

help: ## Show this help message
	@echo "Cline Recorder MCP Server - Available Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

setup: ## Run setup script
	python setup.py

test: ## Run tests
	python test_server.py

run-mcp: ## Start MCP server only
	python main.py --mode mcp

run-api: ## Start REST API server only
	python main.py --mode api

run-both: ## Start both MCP and API servers
	python main.py --mode both

init-db: ## Initialize database tables
	python main.py --init-db

clean: ## Clean up generated files
	rm -f *.db
	rm -rf __pycache__
	rm -rf */__pycache__
	find . -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t cline-recorder .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-stop: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

dev-setup: install setup ## Install dependencies and run setup
	@echo "Development setup complete!"

prod-setup: install ## Install dependencies for production
	@echo "Production setup complete!"
	@echo "Remember to configure your production database and environment variables"