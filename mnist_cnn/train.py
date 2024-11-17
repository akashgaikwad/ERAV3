import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input: 28x28x1
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),    # 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 14x14x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 7x7x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 7x7x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 3x3x128
            
            nn.Flatten(),                                  # 128 * 3 * 3 = 1152
            nn.Linear(1152, 512),                          # Corrected input dimension
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
        # Print model summary
        logger.info(f'Model architecture:')
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f'Total parameters: {total_params:,}')

    def forward(self, x):
        return self.network(x)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
if torch.cuda.is_available():
    logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

# Hyperparameters
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.001

def load_data():
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        drop_last=True,  # Drop the last incomplete batch
        num_workers=2    # Use multiple workers for loading
    )
    
    logger.info(f'Dataset size: {len(train_dataset)}')
    logger.info(f'Number of batches: {len(train_loader)}')
    return train_loader

def save_logs(epochs, losses, accuracies):
    data = {
        'epochs': epochs,
        'losses': losses,
        'accuracies': accuracies
    }
    with open('training_logs.json', 'w') as f:
        json.dump(data, f)
    logger.info(f'Logs saved to training_logs.json')

# Add function to calculate output size
def get_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return ((input_size + 2 * padding - kernel_size) // stride) + 1

def train():
    train_loader = load_data()
    model = CNN().to(device)
    
    # Debug dimensions
    sample_batch = next(iter(train_loader))[0]
    x = sample_batch
    logger.info(f'Input shape: {x.shape}')
    
    x = model.network[0](x)  # First conv
    logger.info(f'After first conv: {x.shape}')
    
    x = model.network[2](x)  # First pool
    logger.info(f'After first pool: {x.shape}')
    
    x = model.network[3](x)  # Second conv
    logger.info(f'After second conv: {x.shape}')
    
    x = model.network[5](x)  # Second pool
    logger.info(f'After second pool: {x.shape}')
    
    x = model.network[6](x)  # Third conv
    logger.info(f'After third conv: {x.shape}')
    
    x = model.network[8](x)  # Third pool
    logger.info(f'After third pool: {x.shape}')
    
    x = model.network[9](x)  # Flatten
    logger.info(f'After flatten: {x.shape}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info('Starting training...')
    epochs_list = []
    losses_list = []
    accuracies_list = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
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
        
        logger.info(f'Epoch {epoch+1}/{EPOCHS}:')
        logger.info(f'Average Loss: {epoch_loss:.4f}')
        logger.info(f'Accuracy: {epoch_acc:.2f}%')
        
        epochs_list.append(epoch + 1)
        losses_list.append(epoch_loss)
        accuracies_list.append(epoch_acc)
        
        save_logs(epochs_list, losses_list, accuracies_list)
    
    logger.info('Training completed!')

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logger.error(f'Training failed: {str(e)}')
        raise