# GR00T eval-server launcher. Wraps docker compose so the venv volume is reused
# instead of leaked. Pass MODEL_PATH (and optionally EMBODIMENT_TAG) as arguments.
#
#   make up      MODEL_PATH=general_3k              # start server (foreground, Ctrl-C stops)
#   make up-d    MODEL_PATH=checkpoints-general_3k  # start server (detached)
#   make down                                       # stop + remove container (keeps venv)
#   make shell                                      # interactive bash in the container
#   make build                                      # (re)build image from docker/Dockerfile
#   make logs                                       # follow logs
#   make clean                                      # remove container + venv volume + prune leaks
#   make prune                                      # reclaim historical dangling 15G volumes

MODEL_PATH     ?= general_3k
EMBODIMENT_TAG ?= NEW_EMBODIMENT
# Newest installed Claude Code native binary from the VS Code extension (survives updates).
CLAUDE_BIN ?= $(shell ls -1t $(HOME)/.vscode-server/extensions/anthropic.claude-code-*-linux-x64/resources/native-binary/claude 2>/dev/null | head -1)
COMPOSE := MODEL_PATH=$(MODEL_PATH) EMBODIMENT_TAG=$(EMBODIMENT_TAG) CLAUDE_BIN=$(CLAUDE_BIN) docker compose

.DEFAULT_GOAL := help
.PHONY: help up up-d down attach claude shell build logs ps seed-vscode clean prune

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Current: MODEL_PATH=$(MODEL_PATH)  EMBODIMENT_TAG=$(EMBODIMENT_TAG)"

up: ## Start the eval server in the foreground (Ctrl-C stops it cleanly)
	$(COMPOSE) up --remove-orphans

up-d: ## Start the eval server detached (then `make attach` or attach VS Code)
	$(COMPOSE) up -d --remove-orphans

down: ## Stop and remove the container (keeps venv + vscode-server volumes)
	$(COMPOSE) down --remove-orphans

attach: ## Attach an interactive shell to the RUNNING container (fast; claude on PATH)
	$(COMPOSE) exec gr00t bash

claude: ## Launch Claude Code inside the running container
	$(COMPOSE) exec gr00t claude

shell: ## Open a shell in a throwaway container (use `attach` for the running one)
	$(COMPOSE) run --rm gr00t bash

build: ## Build the gr00t image from docker/Dockerfile (honours .dockerignore)
	$(COMPOSE) build

logs: ## Follow the running container's logs
	$(COMPOSE) logs -f

ps: ## Show container status
	$(COMPOSE) ps

seed-vscode: ## Pre-load the vscode-server volume from the host so even the FIRST attach is instant
	$(COMPOSE) run --rm -v $(HOME)/.vscode-server:/host-vscode:ro gr00t \
		bash -c 'cp -an /host-vscode/. /root/.vscode-server/ && echo "vscode-server volume seeded"'

clean: ## Full teardown: remove container + venv AND vscode-server volumes, then prune dangling
	$(COMPOSE) down -v --remove-orphans
	docker volume prune -f

prune: ## Reclaim historical dangling anonymous volumes (the leftover ~15G ones)
	docker volume prune -f
