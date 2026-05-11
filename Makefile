.PHONY: build test test-doc lint fmt check deploy

build:
	cargo build

test:
	cargo nextest run --workspace
	cargo test --workspace --doc

test-doc:
	cargo test --workspace --doc

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt --all

check: fmt lint test
	@echo "All checks passed"

deploy:
	cd ~/deploy/krolik-server && \
	docker compose build --no-cache ox-whisper && \
	docker compose up -d --no-deps --force-recreate ox-whisper
