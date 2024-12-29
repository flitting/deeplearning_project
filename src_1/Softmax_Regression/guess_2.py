import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "../../data/"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(
    root=dataset_path,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=dataset_path,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CustomNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-2, lam=1e-4):
        self.lam = lam
        self.lr = lr
        self.weights = [
            torch.randn(input_size, hidden_size, device=device) * 0.01,
            torch.randn(hidden_size, output_size, device=device) * 0.01
        ]
        self.biases = [
            torch.zeros(hidden_size, device=device),
            torch.zeros(output_size, device=device)
        ]
        self.relu = torch.relu

    def forward(self, x):
        self.z1 = x @ self.weights[0] + self.biases[0]
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.weights[1] + self.biases[1]
        return self.z2

    def backward(self, x, y):
        m = x.shape[0]
        dz2 = (self.z2 - y) / m
        dw2 = self.a1.T @ dz2 + self.lam * self.weights[1]
        db2 = dz2.sum(dim=0)
        dz1 = (dz2 @ self.weights[1].T) * (self.z1 > 0).float()
        dw1 = x.T @ dz1 + self.lam * self.weights[0]
        db1 = dz1.sum(dim=0)
        return dw1, db1, dw2, db2

    def update_params(self, dw1, db1, dw2, db2):
        self.weights[0] -= self.lr * dw1
        self.biases[0] -= self.lr * db1
        self.weights[1] -= self.lr * dw2
        self.biases[1] -= self.lr * db2

    def train(self, train_loader, test_loader, epochs=10):
        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.view(-1, 28 * 28).to(device)
                labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
                outputs = self.forward(images)
                dw1, db1, dw2, db2 = self.backward(images, labels)
                self.update_params(dw1, db1, dw2, db2)
            accuracy = self.test(test_loader)
            print(f"Epoch {epoch + 1}, Accuracy: {accuracy}")

    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

network = CustomNetwork(input_size=28 * 28, hidden_size=30, output_size=10)
network.train(train_loader, test_loader, epochs=10)