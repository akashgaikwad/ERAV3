from flask import Flask, render_template, jsonify, make_response, request
import json
import os
import threading
from train import train_model

app = Flask(__name__)
LOG_FILE = 'training_logs.json'

# Global training status
training_status = {
    'is_training': False,
    'current_model': None
}

def initialize_logs():
    """Initialize or reset the logs file"""
    data = {
        'model1': {'config': {}, 'training_params': {}, 'progress': {}, 'epochs': [], 'losses': [], 'accuracies': []},
        'model2': {'config': {}, 'training_params': {}, 'progress': {}, 'epochs': [], 'losses': [], 'accuracies': []}
    }
    with open(LOG_FILE, 'w') as f:
        json.dump(data, f)

@app.route('/')
def home():
    response = make_response(render_template('monitor.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/get_logs')
def get_logs():
    try:
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        print(f"Error reading logs: {e}")
        return jsonify({
            'model1': {'config': {}, 'training_params': {}, 'progress': {}, 'epochs': [], 'losses': [], 'accuracies': []},
            'model2': {'config': {}, 'training_params': {}, 'progress': {}, 'epochs': [], 'losses': [], 'accuracies': []}
        })

@app.route('/train', methods=['POST'])
def train():
    if training_status['is_training']:
        return jsonify({
            'status': 'error',
            'message': f'Training in progress (Currently training {training_status["current_model"]})'
        })
    
    data = request.json
    model1_config = data['model1']
    model2_config = data['model2']
    
    # Initialize logs
    initialize_logs()
    
    def train_both_models():
        training_status['is_training'] = True
        
        # Train model 1
        training_status['current_model'] = 'model1'
        train_model(model1_config, 'model1')
        
        # Train model 2
        training_status['current_model'] = 'model2'
        train_model(model2_config, 'model2')
        
        training_status['is_training'] = False
        training_status['current_model'] = None
    
    thread = threading.Thread(target=train_both_models)
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Training started for both models'
    })

@app.route('/status')
def get_status():
    return jsonify(training_status)

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5001))
    print(f"Starting server on port {PORT}")
    app.run(host='127.0.0.1', port=PORT, debug=True) 