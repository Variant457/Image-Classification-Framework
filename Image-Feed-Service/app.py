import os
import random
import threading
import base64
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, jsonify, request
import requests
from werkzeug.utils import secure_filename

APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "5001"))

# Directory with seed images (cats/dogs).
IMAGES_DIR = os.getenv("IMAGES_DIR", os.path.join(os.path.dirname(__file__), "images"))

# AI service endpoint.
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-classification-service:8000/predict")


app = Flask(__name__)

# In-memory store of last processed result.
_last_lock = threading.Lock()
_last_result: Optional[Dict[str, Any]] = None


def list_images() -> list[str]:
    """Return sorted list of valid image file paths in IMAGES_DIR."""
    if not os.path.isdir(IMAGES_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    files = []
    for name in os.listdir(IMAGES_DIR):
        path = os.path.join(IMAGES_DIR, name)
        _, ext = os.path.splitext(name.lower())
        if os.path.isfile(path) and ext in exts:
            files.append(path)
    return sorted(files)


def choose_image(files: list[str]) -> str:
    """Simulate a feed by randomly selecting an image."""
    return random.choice(files)


def call_ai(image_path: str) -> Dict[str, Any]:
    """Send image to AI container and return {status_code, payload}."""
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "application/octet-stream")}
        r = requests.post(AI_SERVICE_URL, files=files)

    try:
        payload = r.json()
    except Exception:
        payload = {"error": f"AI service returned non-JSON (status {r.status_code})"}

    return {"status_code": r.status_code, "payload": payload}


@app.route("/feed", methods=["POST"])
def feed_once():
    """
    Trigger processing of one image from IMAGES_DIR.
    Dashboard should call this to push a new image through the pipeline.
    """
    files = list_images()
    if not files:
        return jsonify({"error": f"No images found in {IMAGES_DIR}"}), 404

    image_path = choose_image(files)
    ai_resp = call_ai(image_path)

    with open(image_path, 'rb') as img_f:
        b64_str = base64.b64encode(img_f.read()).decode("utf-8")

    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "image_name": os.path.basename(image_path),
        "image_path": image_path,  # internal container path; dashboard can ignore
        "image_base64": b64_str,
        "ai": ai_resp["payload"],
    }

    with _last_lock:
        global _last_result
        _last_result = result

    # If AI errored, propagate a gateway-ish error to dashboard
    code = 200 if ai_resp["status_code"] == 200 else 502
    return jsonify(result), code


@app.route("/last", methods=["GET"])
def last_result():
    """Return last processed result (204 if none yet)."""
    with _last_lock:
        if _last_result is None:
            return jsonify({"message": "No image processed yet"}), 204
        return jsonify(_last_result)


@app.route("/images", methods=["GET"])
def images_list():
    """List available images in pool (debug/helpful for demo)."""
    files = list_images()
    return jsonify({"count": len(files), "images": [os.path.basename(p) for p in files]})


@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Optional helper endpoint:
    lets you add new images to the pool while running.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file'"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs(IMAGES_DIR, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = os.path.join(IMAGES_DIR, filename)
    file.save(save_path)

    return jsonify({"message": "uploaded", "filename": filename}), 201


@app.route("/health", methods=["GET"])
def health():
    """Basic health check for compose/monitoring."""
    return jsonify({"status": "ok", "images_dir": IMAGES_DIR, "ai_url": AI_SERVICE_URL})


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=False)
