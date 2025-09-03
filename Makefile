VENV=.venv
PY=python
PIP=$(VENV)/bin/pip

.PHONY: help venv install link build train eval wf predict status clean

help:
	@echo ""
	@echo "F1 Teammate Qualifying — common commands"
	@echo "  make venv        # create virtual env"
	@echo "  make install     # install deps"
	@echo "  make link        # symlink input data (../f1-ml/data_processed -> data/input)"
	@echo "  make build       # process data -> features"
	@echo "  make train       # train models (+ calibrator if enabled)"
	@echo "  make eval        # evaluate on val/test (with baselines)"
	@echo "  make wf          # walk-forward validation"
	@echo "  make predict EVENT=2025_11  # predict for event"
	@echo "  make status      # show project status + next step"
	@echo "  make clean       # remove caches and temp artifacts"
	@echo ""

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

link:
	@mkdir -p data
	@if [ -d ../f1-ml/data_processed/raw ]; then \
		ln -snf ../f1-ml/data_processed/raw data/input; \
		echo "Linked ../f1-ml/data_processed/raw -> data/input"; \
	else \
		echo "No ../f1-ml/data_processed/raw found. Copy your parquet exports into data/input/"; \
	fi

build:
	$(VENV)/bin/python run_all.py --build

train:
	$(VENV)/bin/python run_all.py --train

eval:
	$(VENV)/bin/python run_all.py --eval

wf:
	$(VENV)/bin/python run_all.py --validate-walkforward

predict:
	@if [ -z "$(EVENT)" ]; then echo "Usage: make predict EVENT=2025_11"; exit 1; fi
	$(VENV)/bin/python run_all.py --predict --event $(EVENT)

status:
	$(VENV)/bin/python run_all.py --status

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +; \
	rm -f reports/train.log || true
