.PHONY: install reproduce numerical test retrain-neural clean

install:
	python -m pip install -r requirements.txt

reproduce:
	python reproduce.py

numerical:
	python reproduce.py --skip-figures

test:
	pytest -q

retrain-neural:
	python -m pip install -r requirements-neural.txt
	python scripts/retrain_neural_models.py

clean:
	rm -rf outputs/tables outputs/figures outputs/results_summary.json outputs/REPRODUCIBILITY_REPORT.md outputs/retrained_neural
	mkdir -p outputs/tables outputs/figures
