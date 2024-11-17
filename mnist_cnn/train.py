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

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info(f'Dataset size: {len(train_dataset)}')
    logger.info(f'Number of batches: {len(train_loader)}')
    return train_loader

def save_logs(model_name, config, training_params, current_epoch, epochs, losses, accuracies):
    """Updated save_logs function to include training parameters and progress"""
    try:
        with open('training_logs.json', 'r') as f:
            data = json.load(f)
    except:
        data = {
            'model1': {'config': {}, 'training_params': {}, 'progress': {}, 'epochs': [], 'losses': [], 'accuracies': []},
            'model2': {'config': {}, 'training_params': {}, 'progress': {}, 'epochs': [], 'losses': [], 'accuracies': []}
        }
    
    data[model_name] = {
        'config': {
            'architecture': config['architecture'],
            'optimizer': config['optimizer']
        },
        'training_params': {
            'batch_size': training_params['batch_size'],
            'total_epochs': training_params['epochs']
        },
        'progress': {
            'current_epoch': current_epoch,
            'epochs_remaining': training_params['epochs'] - current_epoch,
            'completion_percentage': (current_epoch / training_params['epochs']) * 100
        },
        'epochs': epochs,
        'losses': losses,
        'accuracies': accuracies
    }
    
    with open('training_logs.json', 'w') as f:
        json.dump(data, f)
    logger.info(f'Logs updated for {model_name}')

def get_optimizer(optimizer_name, model_parameters, lr=0.001):
    """Get optimizer based on name"""
    optimizers = {
        'adam': lambda: optim.Adam(model_parameters, lr=lr),
        'sgd': lambda: optim.SGD(model_parameters, lr=lr, momentum=0.9),
        'rmsprop': lambda: optim.RMSprop(model_parameters, lr=lr),
        'adamw': lambda: optim.AdamW(model_parameters, lr=lr, weight_decay=0.01)
    }
    return optimizers.get(optimizer_name, optimizers['adam'])()

def train_model(config, model_name, training_params):
    """Updated train_model function with enhanced logging"""
    kernel_config = config['architecture']
    optimizer_name = config['optimizer']
    batch_size = training_params['batch_size']
    total_epochs = training_params['epochs']
    
    logger.info(f'Starting training for {model_name} with:')
    logger.info(f'- Kernels: {kernel_config}')
    logger.info(f'- Optimizer: {optimizer_name}')
    logger.info(f'- Batch Size: {batch_size}')
    logger.info(f'- Total Epochs: {total_epochs}')
    
    model = CNN(kernel_config).to(DEVICE)
    train_loader = load_data(batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters())
    
    epochs_list = []
    losses_list = []
    accuracies_list = []
    
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{total_epochs}')
        
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
                'acc': f'{100.*correct/total:.2f}%',
                'remaining': f'{total_epochs-epoch-1} epochs'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        epochs_list.append(epoch + 1)
        losses_list.append(epoch_loss)
        accuracies_list.append(epoch_acc)
        
        save_logs(model_name, config, training_params, epoch + 1, epochs_list, losses_list, accuracies_list)
    
    logger.info(f'Training completed for {model_name}!')
    return model

if __name__ == '__main__':
    # For testing individual model training
    test_config = [16, 32, 64]
    train_model(test_config, 'model1')