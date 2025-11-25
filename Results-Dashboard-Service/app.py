import os
import requests
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

FEED_URL = os.getenv("FEED_URL", "http://image-feed-service:5001")
AI_URL = os.getenv("AI_URL", "http://ai-classification-service:8000")
TIMEOUT = float(os.getenv("REQ_TIMEOUT", "3.5"))

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/last")
def api_last():
    try:
        r = requests.get(f"{FEED_URL}/last", timeout=TIMEOUT)
        if r.status_code == 204:
            return jsonify({"message": "No image processed yet"}), 200
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": f"Could not reach Feed Service: {e}"}), 502

@app.post("/api/feed")
def api_feed():
    try:
        r = requests.post(f"{FEED_URL}/feed", timeout=TIMEOUT)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": f"Could not trigger Feed Service: {e}"}), 502

@app.get("/api/ai-health")
def api_ai_health():
    # Optional helper for debugging
    try:
        r = requests.get(f"{AI_URL}/health", timeout=TIMEOUT)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": f"Could not reach AI Service: {e}"}), 502


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=False)
