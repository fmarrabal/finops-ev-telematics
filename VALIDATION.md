# Validation record

Validation date: 2026-07-13

The archive was validated in a CPU-only environment with Python 3.13.5 and the
direct dependency versions listed in `requirements-lock.txt`.

Commands executed:

```bash
python reproduce.py
pytest -q
python -m compileall -q src scripts reproduce.py tests
python scripts/retrain_neural_models.py --epochs 1 --patience 1 --mc-draws 2 \
  --output-dir /tmp/finops-neural-smoke
```

Results:

- all numerical reference checks passed;
- 11 automated tests passed;
- all modules compiled successfully;
- the optional PyTorch retraining path completed successfully in smoke-test mode;
- Figures 1-5 were generated and visually inspected.

The full neural retraining command uses the manuscript configuration and is
intentionally separate from the deterministic publication ledger; see the
README for the rationale.
