import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64*7*7)  # Flatten the 64x7x7 feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def test(model, test_loader, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for all batches
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    print(f"Accuracy on test set: {100 * correct / total}%")

    # Calculate precision, recall, F1-score, and support for each class
    report = classification_report(all_labels, all_predicted, target_names=[str(i) for i in range(num_classes)])
    print(report)

def count_instances(dataset):
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return class_counts

# Custom Dataset class to pair images with labels
class CustomMNIST(Dataset):
    def __init__(self, mnist_dataset, new_images, new_label):
        self.mnist_dataset = mnist_dataset
        self.new_images = new_images
        self.new_label = new_label

    def __len__(self):
        return len(self.mnist_dataset) + len(self.new_images)

    def __getitem__(self, idx):
        if idx < len(self.mnist_dataset):
            return self.mnist_dataset[idx]
        else:
            new_idx = idx - len(self.mnist_dataset)
            return self.new_images[new_idx], self.new_label

# Load MNIST dataset
mnist_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)

# Load the images tensor
images = torch.load("images_acd_1.pt", map_location=torch.device('cpu'))

# Split images into train and test sets
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

# Load the test dataset
test_dataset = MNIST(root='./data', train=False, transform=ToTensor())

# Create a new dataset with the images tensor and label 10 for train and test datasets
new_label = 10
train_dataset = CustomMNIST(mnist_dataset, train_images, new_label)
test_dataset = CustomMNIST(test_dataset, test_images, new_label)

# Count instances in the datasets
train_dataset_counts = count_instances(train_dataset)
test_dataset_counts = count_instances(test_dataset)

# Output instance counts
print("\nTrain Dataset after:")
for class_label, count in train_dataset_counts.items():
    print(f"Class {class_label}: Instances: {count}")

print("\nTest Dataset after:")
for class_label, count in test_dataset_counts.items():
    print(f"Class {class_label}: Instances: {count}")


# Hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 5

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the model
train(model, train_loader, optimizer, criterion, epochs=epochs)

# Test the model
test(model, test_loader, 11)
