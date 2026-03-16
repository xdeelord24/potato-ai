"""
Optional HTTP API server for Potato AI inference.
Lightweight Flask/FastAPI - for local deployment on potato computers.
"""

from pathlib import Path
from typing import List, Optional

# Minimal server - no heavy deps required
try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


def create_server(
    model_path: Optional[Path] = None,
    config=None,
    host: str = "127.0.0.1",
    port: int = 5000,
):
    """Create inference API server."""
    if not HAS_FLASK:
        raise ImportError("Install flask: pip install flask")

    from inference.engine import InferenceEngine
    from src.ssm import PotatoConfig, PotatoLM

    if config is None:
        config = PotatoConfig.tiny()

    if model_path and model_path.exists():
        engine = InferenceEngine(config=config, checkpoint_path=model_path)
    else:
        model = PotatoLM(config)
        engine = InferenceEngine(model=model)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/complete", methods=["POST"])
    def complete():
        data = request.get_json() or {}
        prompt = data.get("prompt", "")
        prompt_ids = data.get("prompt_ids", [])
        max_new_tokens = data.get("max_new_tokens", 64)
        temperature = data.get("temperature", 0.8)

        if prompt_ids:
            ids = prompt_ids
        elif prompt:
            # Simple char-level encoding if no tokenizer
            ids = [ord(c) % 32000 for c in prompt[:512]]
        else:
            return jsonify({"error": "Provide prompt or prompt_ids"}), 400

        result = engine.complete(
            prompt_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return jsonify({"tokens": result})

    return app
