.PHONY: test
test:
	pytest -r test/

.PHONY: lint
lint:
	black --check src/ tests/
	ruff check src/ tests/
	pyright src/ tests/

.PHONY: docs
docs:
	(cd docs && make html)
