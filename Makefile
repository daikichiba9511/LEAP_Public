.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash
COMPE := leap-atmospheric-physics-ai-climsim
PYTHONPATH := $(shell pwd)

.PHONY: setup
setup: ## setup install packages
	@python -m pip install --upgrade pip setuptools wheel
	@python -m pip install -e .
	@python -m pip install -e .[dev]

.PHONY: download_data
download_data: ## download data from competition page
	@kaggle competitions download -c "${COMPE}" -p ./input;
	@unzip "./input/${COMPE}.zip" -d "./input/${COMPE}"

.PHONY: upload
upload: ## upload dataset
	@kaggle datasets version --dir-mode zip -p ./src -m "update $(date +'%Y-%m-%d %H:%M:%S')"
	@kaggle datasets version --dir-mode zip -p ./output/submit -m "update $(date +'%Y-%m-%d %H:%M:%S')"

.PHONY: lint
lint: ## lint code
	@ruff check scripts src

.PHONY: mypy
mypy: ## typing check
	@mypy --config-file pyproject.toml scirpts src

.PHONY: fmt
fmt: ## auto format
	@ruff check --fix scripts src
	@ruff format scripts src

.PHONY: test
test: ## run test with pytest
	@pytest -c tests

.PHONY: setup-dev
setup-dev: ## setup my dev env by installing my dotfiles
	git clone git@github.com:daikichiba9511/dotfiles.git ~/dotfiles && cd ~/dotfiles && bash setup.sh && cd -

.PHONY: clean
clean: ## clean outputs
	@rm -rf ./output/*
	@rm -rf ./wandb
	@rm -rf ./debug
	@rm -rf ./.venv

%:
	@echo 'command "$@" is not found.'
	@$(MAKE) help
	@exit 1

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
