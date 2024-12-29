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

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

train_images, train_labels = next(iter(train_loader))
test_images, test_labels = next(iter(test_loader))

train_images = torch.reshape(train_images, (len(train_images), -1)).to(device)
test_images = torch.reshape(test_images, (len(test_images), -1)).to(device)
print(train_images[0])
train_labels_one_hot = torch.nn.functional.one_hot(train_labels).float().to(device)
test_labels_one_hot = torch.nn.functional.one_hot(test_labels).float().to(device)

label_num = 10


class CustomNetwork:
    def __init__(self, param_train_images, param_train_labels, param_label_num=10, lr=1e-2):
        """

        :param param_train_images: should be a transposed mathematics matrix
        :param param_train_labels: should be a transposed one hot mathematics matrix
        :param param_label_num:
        """
        self.lam = 0.01
        self.train_images = param_train_images
        self.train_labels = param_train_labels
        self.label_num = param_label_num
        self.layer_num = 4
        self.lr = lr
        self.batch_size = train_images.shape[0]
        node_nums = (train_images.shape[-1], 30, 20, param_label_num)
        self.weights = []
        self.biases = []
        for i in range(1, len(node_nums)):
            weight = torch.randn(node_nums[i - 1], node_nums[i], device=device)
            bias = torch.randn(node_nums[i], device=device)
            self.weights.append(weight)
            self.biases.append(bias)

        self.func = torch.relu
        self.d_func = lambda y: (y > 0).float()
        # torch.tensor -> torch.tensor
        self.residuals = [torch.empty(0, device=device) for _ in range(self.layer_num)]
        self.activations = [torch.empty(0, device=device) for _ in range(self.layer_num)]
        self.deltas = [torch.empty(0, device=device) for _ in range(self.layer_num)]
        self.grads = [torch.empty(0, device=device) for _ in range(self.layer_num - 1)]
        self.grad_biases = [torch.empty(0, device=device) for _ in range(self.layer_num - 1)]
        self.weight_changes = [[] for _ in range(4)]

    def forward(self, data: torch.Tensor):
        self.activations[0] = data
        for i in range(0, self.layer_num - 1):
            self.residuals[i + 1] = self.activations[i] @ self.weights[i] + self.biases[i]
            self.residuals[i + 1] = torch.clamp(self.residuals[i + 1], min=-10, max=10)
            self.activations[i + 1] = self.func(self.residuals[i + 1])

    def backward(self, labels: torch.Tensor):
        self.deltas[-1] = -1 * (labels - self.activations[-1]) * self.d_func(self.residuals[-1])
        for i in range(self.layer_num - 2, 0, -1):
            self.deltas[i] = (self.deltas[i + 1] @ self.weights[i].T) * self.d_func(self.residuals[i])

        for i in range(0, self.layer_num - 1):
            self.grads[i] = self.activations[i].T @ self.deltas[i + 1]
            self.grad_biases[i] = self.deltas[i + 1].mean(dim=0)
            self.weight_changes[i] = [-1 * self.lr * (self.grads[i] + self.lam * self.weights[i]) / self.batch_size,
                                      -1 * self.lr * (self.grad_biases[i] + self.biases[i]) / self.batch_size]

    def update_params(self):
        for i in range(0, self.layer_num - 1):
            self.weights[i] = self.weights[i] - self.lr * (self.grads[i] + self.lam * self.weights[i]) / self.batch_size
            self.biases[i] = self.biases[i] - self.lr * (self.grad_biases[i] + self.biases[i]) / self.batch_size

    def train(self, epochs=50):
        for epoch in range(epochs):
            self.forward(self.train_images)
            self.backward(self.train_labels)
            self.update_params()
            if self.lr > 1e-7:
                self.lr = self.lr * 0.9
            accuracy = self.test(test_images, test_labels_one_hot)
            print(f"Epoch {epoch + 1}, Accuracy: {accuracy.item()}")

    def test(self, param_test_images, param_test_labels):
        self.forward(param_test_images)
        result = torch.argmax(self.activations[-1], dim=-1) == torch.argmax(param_test_labels, dim=-1)
        return result.float().mean()


network = CustomNetwork(train_images, train_labels_one_hot, label_num)
network.train(epochs=100)
