import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import logging
from torchvision import datasets, transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MPS (Metal Performance Shaders) Setup for Apple Silicon
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using Apple Metal GPU")
else:
    DEVICE = torch.device("cpu")
    logger.warning("MPS is not available. Using CPU instead.")

# Hyperparameters
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 2  # MPS works better with fewer workers

class CNN(nn.Module):
    def __init__(self, kernel_config):
        super(CNN, self).__init__()
        k1, k2, k3 = kernel_config
        
        self.network = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(k1, k2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(k2, k3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(k3 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
        # Print model summary
        logger.info(f'Model architecture with kernels {kernel_config}:')
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f'Total parameters: {total_params:,}')

    def forward(self, x):
        return self.network(x)

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info(f'Dataset size: {len(train_dataset)}')
    logger.info(f'Number of batches: {len(train_loader)}')
    return train_loader

def save_logs(model_name, epochs, losses, accuracies):
    try:
        with open('training_logs.json', 'r') as f:
            data = json.load(f)
    except:
        data = {
            'model1': {'epochs': [], 'losses': [], 'accuracies': []},
            'model2': {'epochs': [], 'losses': [], 'accuracies': []}
        }
    
    data[model_name] = {
        'epochs': epochs,
        'losses': losses,
        'accuracies': accuracies
    }
    
    with open('training_logs.json', 'w') as f:
        json.dump(data, f)
    logger.info(f'Logs updated for {model_name}')

def train_model(kernel_config, model_name):
    logger.info(f'Starting training for {model_name} with kernels {kernel_config}')
    
    # Initialize model and move to MPS
    model = CNN(kernel_config).to(DEVICE)
    
    train_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    epochs_list = []
    losses_list = []
    accuracies_list = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{EPOCHS}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to MPS
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            running_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        logger.info(f'{model_name} Epoch {epoch+1}/{EPOCHS}:')
        logger.info(f'Average Loss: {epoch_loss:.4f}')
        logger.info(f'Accuracy: {epoch_acc:.2f}%')
        
        epochs_list.append(epoch + 1)
        losses_list.append(epoch_loss)
        accuracies_list.append(epoch_acc)
        
        save_logs(model_name, epochs_list, losses_list, accuracies_list)
    
    logger.info(f'Training completed for {model_name}!')
    return model

if __name__ == '__main__':
    # For testing individual model training
    test_config = [16, 32, 64]
    train_model(test_config, 'model1')