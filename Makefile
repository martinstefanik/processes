build:
	poetry build

clean:
	rm -rf dist htmlcov .coverage .mypy_cache .pytest_cache

test:
	poetry run pytest tests \
		--exitfirst \
		--numprocesses auto \
		--cov=processes \
		--cov-report=html
