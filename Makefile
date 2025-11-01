.PHONY: help install install-dev lint format test test-unit test-integration test-e2e clean up down logs build shell migrate db-upgrade db-downgrade pre-commit

help:
	@echo "ResearcherAI - Makefile Commands"
	@echo "================================="
	@echo "Development:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo "  make lint            Run linting (ruff)"
	@echo "  make format          Format code (ruff format, isort)"
	@echo "  make pre-commit      Run pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests"
	@echo "  make test-integration Run integration tests"
	@echo "  make test-e2e        Run end-to-end tests"
	@echo ""
	@echo "Docker:"
	@echo "  make up              Start all services"
	@echo "  make down            Stop all services"
	@echo "  make build           Build Docker images"
	@echo "  make logs            Show logs"
	@echo "  make shell           Open shell in app container"
	@echo ""
	@echo "Database:"
	@echo "  make migrate         Create new migration"
	@echo "  make db-upgrade      Apply migrations"
	@echo "  make db-downgrade    Rollback migration"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove caches and artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	@echo "Running Ruff linter..."
	ruff check src/ tests/ --fix
	@echo "Running type checks..."
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting with Ruff..."
	ruff format src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/

test:
	@echo "Running all tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-e2e:
	@echo "Running end-to-end tests..."
	pytest tests/e2e/ -v

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "Cleanup complete!"

up:
	@echo "Starting services..."
	docker compose up -d
	@echo "Services started. Check status with: docker compose ps"

down:
	@echo "Stopping services..."
	docker compose down

build:
	@echo "Building Docker images..."
	docker compose build

logs:
	docker compose logs -f

shell:
	docker compose exec rag-multiagent /bin/bash

migrate:
	@echo "Creating new migration..."
	alembic revision --autogenerate -m "$(msg)"

db-upgrade:
	@echo "Applying migrations..."
	alembic upgrade head

db-downgrade:
	@echo "Rolling back migration..."
	alembic downgrade -1

pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files
