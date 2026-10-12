"""Serve ej predictions from a small FastAPI application.

Install the example-only dependencies before starting the server::

    pip install fastapi uvicorn
    uvicorn examples.fastapi_server:app --reload

Then send a JSON list in the same record format used by the README::

    curl -X POST http://127.0.0.1:8000/predict \
      -H 'content-type: application/json' \
      -d '[{"state":"The parcel arrived late.","questions":{"needs_human":{"type":"noul","instructions":"A human should review this request.","options":[{"key":"false","text":"No"},{"key":"true","text":"Yes"}]}}}]'

FastAPI and Uvicorn are intentionally not package dependencies. The model is
loaded once when this module is imported, so requests reuse the same model.
Set ``EJ_WEIGHTS`` and ``EJ_REVISION`` to use a different model source.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

import ej


WEIGHTS = os.environ.get("EJ_WEIGHTS", "5ak3t/ej")
REVISION = os.environ.get("EJ_REVISION", "v0.0.1")

model = ej.load(WEIGHTS, revision=REVISION)
app = FastAPI(title="ej prediction example")


@app.post("/predict")
def predict(records: list[dict[str, Any]]) -> list[dict[str, list[float]]]:
    """Return one probability distribution per question in each record."""
    return model.predict(records)
