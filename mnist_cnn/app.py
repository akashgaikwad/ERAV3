from flask import Flask, render_template, jsonify, make_response
import json
import os

app = Flask(__name__)
LOG_FILE = 'training_logs.json'

@app.route('/')
def home():
    # Prevent caching of the main page
    response = make_response(render_template('monitor.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/get_logs')
def get_logs():
    try:
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        # Prevent caching of the JSON data
        response = make_response(jsonify(data))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    except Exception as e:
        print(f"Error reading logs: {e}")  # Debug log
        return jsonify({'epochs': [], 'losses': [], 'accuracies': []})

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5001))
    print(f"Starting server on port {PORT}")
    app.run(host='127.0.0.1', port=PORT, debug=True) 