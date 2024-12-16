from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root="../../data/",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.MNIST(
    root="../../data/",
    train=False,
    download=True,
    transform=transform
)


def filter_by_label(data):
    image, label = data
    return label == 0 or label == 1


# 筛选标签为 0 或 1 的索引
train_indices = (train_data.targets == 0) | (train_data.targets == 1)
test_indices = (test_data.targets == 0) | (test_data.targets == 1)

# 创建子集
train_data = Subset(train_data, torch.where(train_indices)[0])
test_data = Subset(test_data, torch.where(test_indices)[0])

train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
train_images, train_labels = next(iter(train_dataloader))
pixel_num = 28 * 28

train_images = torch.reshape(train_images, (len(train_images), pixel_num))
train_labels = torch.reshape(train_labels, (len(train_images),))

test_images, test_labels = next(iter(DataLoader(test_data, batch_size=len(train_data), shuffle=False)))
test_images = torch.reshape(test_images, (len(test_images), pixel_num))
test_labels = torch.reshape(test_labels, (len(test_images),))


class CustomNetwork:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, learning_rate: float):
        self.X = X
        self.X = torch.cat(
            (
                X, (torch.ones(X.shape[0], 1))
            ), dim=1
        )
        self.Y = Y.float()
        self.W = torch.rand(self.X.shape[-1], )
        self.learning_rate = learning_rate
        self.loss_history = []

    def train(self):
        Y_hat = torch.sigmoid(torch.matmul(self.W, self.X.transpose(0, 1)))
        nabla = torch.matmul(Y_hat - self.Y, self.X)
        self.W = self.W - self.learning_rate * nabla
        loss = self.get_loss(Y_hat)
        self.loss_history.append(loss)
        """
        distance = self.gradient_check(self.W, self.X, nabla, )
        print(f"distance:{distance}")
        """
        return loss

    def test(self, X_test, Y_test):
        X_test = torch.cat(
            (
                X_test, (torch.ones(X_test.shape[0], 1))
            ), dim=1
        )
        Y_hat = torch.sigmoid(torch.matmul(self.W, X_test.transpose(0, 1)))
        # noinspection PyUnresolvedReferences
        Y_hat = (Y_hat > 0.5).int()
        res = torch.mean(torch.abs(Y_test - Y_hat).float())
        return 1 - res.item()

    def get_loss(self, Y_hat):
        epsilon = 1e-7  # A small value to prevent log(0)
        Y_hat = torch.clamp(Y_hat, min=epsilon, max=1 - epsilon)  # Clamp the predictions to avoid log(0)
        Y = torch.clamp(self.Y, min=epsilon, max=1 - epsilon)
        return -1 * Y @ torch.log(Y_hat) - (1 - Y) @ torch.log(1 - Y_hat)

    def _loss(self, w: torch.Tensor):
        Y_hat = torch.sigmoid(torch.matmul(self.X, w))
        return self.get_loss(Y_hat)

    def gradient_check(self, w: torch.Tensor, X, nabla: torch.Tensor):
        feature_num = len(w)
        W = w.repeat((feature_num, 1))
        epsilon = 1e-5
        Epsilon = torch.eye(feature_num) * epsilon

        def L(Thetas: torch.Tensor, f: Callable):
            thetas = Thetas.unbind(0)
            return torch.tensor(list(map(f, thetas)))

        grad = (L(W + Epsilon, self._loss) - L(W - Epsilon, self._loss)) / (2 * epsilon)
        difference_vector = grad - nabla

        norm_difference = torch.norm(difference_vector) / ((torch.norm(grad) + torch.norm(nabla)) / 2)
        return norm_difference


nn = CustomNetwork(train_images, train_labels, 0.0001)
epoch = 100
for i in range(epoch):
    epoch_loss = nn.train()
    print(f"training for epoch {i},loss:{epoch_loss}")
correct_rate = nn.test(test_images, test_labels)
print(correct_rate)
# Assuming nn.loss_history contains the loss values for each epoch
losses = nn.loss_history

# Plot loss curve with logarithmic y-axis
plt.figure(figsize=(10, 6))
plt.plot(range(1, epoch + 1), losses, label="Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.yscale('log')  # Set the y-axis to log scale
plt.title("Loss vs. Epoch")
plt.legend()
plt.show()
