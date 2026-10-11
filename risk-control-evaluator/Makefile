.PHONY: install install-dev test lint format type-check clean run-web run-cli help

PYTHON := python
PIP := pip

help:
	@echo "Available targets:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run pytest with coverage"
	@echo "  lint         Run ruff linter"
	@echo "  format       Format code with ruff"
	@echo "  type-check   Run mypy type checker"
	@echo "  check        Run lint + type-check + test"
	@echo "  clean        Clean build artifacts"
	@echo "  run-web      Launch Streamlit web app"
	@echo "  run-cli      Show CLI help"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[all]"

test:
	pytest -v

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

type-check:
	mypy src/risk_eval

check: lint type-check test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-web:
	streamlit run src/risk_eval/web/app.py

run-cli:
	risk-eval --help
