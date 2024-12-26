from typing import Callable

import torch
from torch.utils.data import DataLoader
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

train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
train_images, train_labels = next(iter(train_dataloader))
pixel_num = 28 * 28
train_images = torch.reshape(train_images, (len(train_images), pixel_num))
train_labels = torch.reshape(train_labels, (len(train_images),))

test_images, test_labels = next(iter(DataLoader(test_data, batch_size=len(train_data), shuffle=False)))
test_images = torch.reshape(test_images, (len(test_images), pixel_num))
test_labels = torch.reshape(test_labels, (len(test_images),))
label_num = len(train_data.targets.unique())


class CustomNetwork:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, learning_rate: float, label_num):
        self.X = torch.cat(
            (
                X, (torch.ones(X.shape[0], 1))
            ), dim=1
        )
        self.Y = torch.nn.functional.one_hot(Y, label_num)
        self.W = torch.randn(label_num, self.X.shape[-1]) * (1.0 / self.X.shape[-1] ** 0.5)

        self.learning_rate = learning_rate
        self.loss_history = []

    def train(self):
        multi_tensor = self.W @ self.X.transpose(0, 1)
        multi_tensor = multi_tensor - torch.max(multi_tensor, dim=0, keepdim=True)[0]  # 减去最大值，防止溢出
        Y_hat = torch.exp(multi_tensor) / torch.sum(torch.exp(multi_tensor), dim=0)

        nabla = torch.matmul((Y_hat - self.Y.t()), self.X)
        #nabla = torch.clamp(nabla, min=-1.0, max=1.0)
        self.W = self.W - self.learning_rate * nabla
        loss = self.get_loss(Y_hat)
        self.loss_history.append(loss)
        print(f"Gradient mean: {nabla.mean().item()}, std: {nabla.std().item()}")
        if self.learning_rate > 1e-10:
            self.learning_rate *= 0.9

        return loss

    def test(self, X_test, Y_test):
        X_test = torch.cat(
            (
                X_test, (torch.ones(X_test.shape[0], 1))
            ), dim=1
        )

        multi_tensor: torch.Tensor = self.W @ X_test.t()
        Y_hat = torch.exp(multi_tensor) / (torch.sum(multi_tensor.exp(), dim=0))
        # noinspection PyUnresolvedReferences
        predicted_labels = Y_hat.argmax(dim=0)
        res = torch.sum(predicted_labels == Y_test) / len(Y_test)
        return res.item()

    def get_loss(self, Y_hat):
        epsilon = 1e-7  # 防止 log(0)
        Y_hat = torch.clamp(Y_hat, min=epsilon, max=1 - epsilon)  # 限制 Y_hat 范围
        loss = -torch.sum(self.Y.t() * torch.log(Y_hat)) / Y_hat.shape[1]
        return loss


nn = CustomNetwork(train_images, train_labels, 1e-5, label_num)
epoch = 200
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
