# flask_app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

def run_app():
    app.run(host='0.0.0.0', port=8000)
