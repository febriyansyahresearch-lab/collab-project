.PHONY: setup test train-nlp train-image train-timeseries train-recommender lint clean

setup:
	pip install -r requirements.txt

test:
	python -m pytest projects/ -v

train-nlp:
	python -m projects.nlp.src.train

train-image:
	python -m projects.image.src.train

train-timeseries:
	python -m projects.timeseries.src.train

train-recommender:
	python -m projects.recommender.src.recommend

lint:
	ruff check . --ignore E501 || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
