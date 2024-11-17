from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)
LOG_FILE = 'training_logs.json'

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/get_logs')
def get_logs():
    try:
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({'epochs': [], 'losses': [], 'accuracies': []})

if __name__ == '__main__':
    app.run(debug=True) 