import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from model import SimpleCNN
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def save_logs(epochs, losses, accuracies):
    data = {
        'epochs': epochs,
        'loss': losses,
        'accuracy': accuracies
    }
    with open('training_log.json', 'w') as f:
        json.dump(data, f)

def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training tracking
    epochs, losses, accuracies = [], [], []

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        epochs.append(epoch + 1)
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        save_logs(epochs, losses, accuracies)

    print('Training finished!')
    torch.save(model.state_dict(), 'mnist_cnn.pth')

    # Show results on random test images
    model.eval()
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        idx = torch.randint(len(test_dataset), (1,)).item()
        img, label = test_dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).item()
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'True: {label}\nPred: {pred}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()

if __name__ == '__main__':
    train() 