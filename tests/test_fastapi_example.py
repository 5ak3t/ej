"""Tests for the optional FastAPI example."""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("torch")

import ej


def test_fastapi_example_returns_422_for_bad_record(monkeypatch) -> None:
    fastapi = pytest.importorskip("fastapi")

    class _Model:
        @staticmethod
        def predict(records):
            raise ej.RecordError("questions must be a non-empty dict")

    module_name = "examples.fastapi_server"
    monkeypatch.setattr(ej, "load", lambda *args, **kwargs: _Model())
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    with pytest.raises(fastapi.HTTPException) as caught:
        module.predict([{"state": "bad"}])

    assert caught.value.status_code == 422
    assert caught.value.detail == "questions must be a non-empty dict"
