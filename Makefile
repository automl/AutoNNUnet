.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install check format
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
ruff: ## run ruff as a formatter
	python -m ruff check --exit-zero autonnunet
	python -m ruff check --silent --exit-zero --no-cache --fix autonnunet

test: ## run tests quickly with the default Python
	python -m pytest tests
cov-report:
	coverage html -d coverage_html

coverage: ## check code coverage quickly with the default Python
	coverage run --source autonnunet -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/autonnunet.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ autonnunet
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

setup-submodules:
	git submodule update --init --recursive

	cd submodules/batchgenerators && git checkout master && git pull origin master && pip install .
	cd submodules/hypersweeper && git checkout dev && git pull origin dev && pip install -e .
	cd submodules/MedSAM && git checkout MedSAM2 && git pull origin MedSAM2
	cd submodules/neps && git checkout master && git pull origin master && pip install .
	cd submodules/nnUNet && git checkout dev && git pull origin dev && pip install .


install: clean setup-submodules ## install the package to the active Python's site-packages
	pip install -e ".[dev]"

check:
	pre-commit run --all-files

format:
	make ruff