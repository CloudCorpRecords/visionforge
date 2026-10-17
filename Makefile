.PHONY: setup serve train test frontend-install frontend-dev frontend-build

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

serve:
	.venv/bin/python visionforge.py serve

train:
	.venv/bin/python visionforge.py train --epochs 2

test:
	.venv/bin/python -m pytest -q

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build
