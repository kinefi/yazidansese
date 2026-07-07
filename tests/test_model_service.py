import sys

from app import model_service


def test_ensure_tokenizer_runtime_installs_when_missing(monkeypatch):
    attempted = []

    def fake_import_module(name):
        raise ImportError(f"missing {name}")

    def fake_check_call(command):
        attempted.append(command)

    monkeypatch.setattr(model_service.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(model_service.subprocess, "check_call", fake_check_call)

    model_service._ensure_tokenizer_runtime()

    assert attempted == [[sys.executable, "-m", "pip", "install", "--quiet", "sentencepiece", "tiktoken"]]
