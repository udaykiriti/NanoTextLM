.PHONY: install prepare train demo infer clean docker-build

install:
	pip install -r requirements.txt

prepare:
	python scripts/prepare_shakespeare.py

train:
	python src/train.py

demo:
	python src/train.py --demo

infer:
	python src/inference.py

web:
	python src/app.py

clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

test:
	pytest tests/

docker-build:
	docker build -t nanotextlm .
