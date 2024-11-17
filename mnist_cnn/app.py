from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    try:
        with open('training_log.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({'epochs': [], 'loss': [], 'accuracy': []})

if __name__ == '__main__':
    app.run(debug=True) 