# Import necessary libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# Set device (GPU or CPU)
print('cuda mode' if torch.cuda.is_available() else 'cpu mode')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Load MNIST dataset


def load_mnist_dataset():
    """Loads MNIST dataset using PyTorch."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Define the neural network model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(12 * 12 * 20, 10)  # input layer (28x28 images) -> output layer (10 classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(-1, 12 * 12 * 20)  # flatten the input
        x = torch.sigmoid(self.fc1(x))
        return x

# Train the model


def train_model(model, train_loader, test_loader):
    """Trains the model on the training data."""
    criterion = nn.MSELoss()  # use mean squared error loss for sigmoid output
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # one-hot encode labels
            one_hot_labels = torch.zeros(labels.size(0), 10).to(device)
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
        model.train()
        print(f'Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}')

# Main function


def main():
    # Load MNIST dataset
    train_loader, test_loader = load_mnist_dataset()

    # Create the model
    model = Net().to(device)

    # Train the model
    train_start = time.time()
    train_model(model, train_loader, test_loader)
    train_end = time.time()
    print(f"training time: {train_end - train_start}")


if __name__ == "__main__":
    main()
