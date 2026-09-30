.PHONY: install install-full test cov demo lint clean

install:            ## core install (numpy, sklearn, pydantic) + dev tools
	pip install -e ".[dev,faiss,docs]"

install-full:       ## adds torch + transformers for Contriever / Llama
	pip install -e ".[all]"

test:
	pytest -q

cov:
	pytest -q --cov=trace_rag --cov-report=term-missing

demo:
	python scripts/demo_end_to_end.py --root runs/demo

clean:
	rm -rf runs .pytest_cache .coverage **/__pycache__ src/*.egg-info
