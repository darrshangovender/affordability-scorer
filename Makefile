.PHONY: install test lint bench data clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -q

lint:
	ruff check .

bench:
	python eval/run.py

data:
	python demo/generate.py

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info .ruff_cache
