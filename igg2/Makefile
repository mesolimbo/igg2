IMAGE   ?= igg2:local
DATA    ?= $(PWD)/data
INPUT   ?= input.csv
MODEL   ?= model.json
ARGS    ?=

DOCKER_RUN  = MSYS_NO_PATHCONV=1 docker run --rm -v "$(DATA):/data" -w /data $(IMAGE)
DOCKER_TOOL = MSYS_NO_PATHCONV=1 docker run --rm -v "$(PWD):/app" -w /app --entrypoint "" $(IMAGE)

.PHONY: help build lock test lint clean train generate inspect shell

help:  ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build the Docker image
	docker build -t $(IMAGE) .

lock: build  ## Regenerate Pipfile.lock inside the container
	$(DOCKER_TOOL) pipenv lock

test: build  ## Run the test suite inside the container
	$(DOCKER_TOOL) pytest test/ -v

lint: build  ## Static checks inside the container
	$(DOCKER_TOOL) python -m compileall src/

clean:  ## Remove build artefacts and caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info

train: build  ## Train: make train INPUT=foo.csv MODEL=foo.json
	$(DOCKER_RUN) train $(INPUT) $(MODEL) $(ARGS)

generate: build  ## Generate: make generate MODEL=foo.json ARGS="--count 20"
	$(DOCKER_RUN) generate $(MODEL) $(ARGS)

inspect: build  ## Inspect a model: make inspect MODEL=foo.json
	$(DOCKER_RUN) inspect $(MODEL)

shell: build  ## Drop into a shell inside the image
	docker run --rm -it -v "$(PWD):/app" -w /app --entrypoint /bin/bash $(IMAGE)
