# embedding-lab

Embedding retrieval choices measured: pooling, similarity, ANN, domain-shift fine-tuning, and chunking.

## Evidence

## What is here

## Run it
```console
uv sync
uv run pytest
uv run el --help
```

## Charts
`docs/views/index.html` draws the results above from the same seeded runs. Rebuild it with
`uv run python scripts/build_views.py`.

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py` enforces both.

## Limits
