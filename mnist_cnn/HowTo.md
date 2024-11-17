# MNIST CNN Project

## Setup Instructions

1. Create project directory and virtual environment:
    bash
    mkdir mnist_cnn
    cd mnist_cnn
    python3 -m venv venv
    source venv/bin/activate

2. Install PyTorch and other requirements:
    bash
    pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    pip3 install flask matplotlib


## Running the Project

1. Start the Flask server (keep this terminal open):
    bash
    python3 app.py

2. In a new terminal, activate venv and start training:
    bash
    cd mnist_cnn
    source venv/bin/activate
    python3 train.py

3. Open your browser and go to:
    http://localhost:5000


## Project Structure
- `model.py`: CNN architecture
- `train.py`: Training script
- `app.py`: Flask server for monitoring
- `templates/index.html`: Web interface

## Results

After training completes:
- Model is saved as `mnist_cnn.pth`
- Test results on 10 random images are saved as `results.png`
- Training logs are in `training_log.json`

## Troubleshooting

If you encounter any installation issues:
1. Make sure you're in the virtual environment (you should see `(venv)` in your terminal)
2. Make sure you're using Python 3
3. If needed, upgrade pip: `pip3 install --upgrade pip`

## Deactivating Virtual Environment

When finished:
```bash
deactivate
```


