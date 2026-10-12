"""Serve ej predictions from a small FastAPI application.

Install the example-only dependencies before starting the server::

    pip install fastapi uvicorn
    uvicorn examples.fastapi_server:app --reload

Then send a JSON list in the same record format used by the README::

    curl -X POST http://127.0.0.1:8000/predict \
      -H 'content-type: application/json' \
      -d '[{"state":"The parcel arrived late.","questions":{"route":{"type":"choice","instructions":"Which team should handle this ticket?","options":[{"key":"billing","text":"Billing"},{"key":"returns","text":"Returns"},{"key":"tech","text":"Tech support"}]}}}]'

FastAPI and Uvicorn are intentionally not package dependencies. The model is
loaded once when this module is imported, so requests reuse the same model.
Set ``EJ_WEIGHTS`` and ``EJ_REVISION`` to use a different model source.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException

import ej

WEIGHTS = os.environ.get("EJ_WEIGHTS", "5ak3t/ej")
REVISION = os.environ.get("EJ_REVISION", "v0.0.1")

model = ej.load(WEIGHTS, revision=REVISION)
app = FastAPI(title="ej prediction example")


@app.post("/predict")
def predict(records: list[dict[str, Any]]) -> list[dict[str, list[float]]]:
    """Return one probability distribution per question in each record."""
    try:
        return model.predict(records)
    except ej.RecordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
