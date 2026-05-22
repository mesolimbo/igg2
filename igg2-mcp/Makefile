IMAGE      ?= igg2-mcp:local
IGG2_IMAGE ?= igg2:local
IGG2_DIR   ?= ../igg2
DATA       ?= $(PWD)/data

DOCKER_TOOL = MSYS_NO_PATHCONV=1 docker run --rm -v "$(PWD):/app" -w /app --entrypoint "" $(IGG2_IMAGE)

.PHONY: help igg2-image build lock test serve clean

help:  ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

igg2-image:  ## Build the base igg2 image this server extends
	MSYS_NO_PATHCONV=1 docker build -t $(IGG2_IMAGE) $(IGG2_DIR)

build: igg2-image  ## Build the MCP server image
	MSYS_NO_PATHCONV=1 docker build -t $(IMAGE) .

lock: igg2-image  ## Regenerate Pipfile.lock inside a container
	$(DOCKER_TOOL) pipenv lock

test: build  ## Run the test suite inside the container
	MSYS_NO_PATHCONV=1 docker run --rm --entrypoint "" $(IMAGE) pytest test/ -v

serve: build  ## Run the MCP server over stdio (as an MCP client would)
	MSYS_NO_PATHCONV=1 docker run -i --rm -v "$(DATA):/data" $(IMAGE)

clean:  ## Remove caches and lock artefacts
	rm -rf .pytest_cache .cache **/__pycache__
