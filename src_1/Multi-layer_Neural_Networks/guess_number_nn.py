import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

dataset_path = "../../data/"

train_dataset = datasets.MNIST(
    root=dataset_path,
    train=True,
    download=False,
)

test_dataset = datasets.MNIST(
    root=dataset_path,
    train=False,
    download=False
)

train_images, train_labels = next(iter(DataLoader(train_dataset)))
test_images, test_labels = next(iter(DataLoader(test_dataset)))

train_images = torch.reshape(train_images, (len(train_images), -1))
test_images = torch.reshape(test_images, (len(test_images), -1))

train_labels_one_hot = torch.nn.functional.one_hot(train_labels)
test_labels_one_hot = torch.nn.functional.one_hot(test_labels)

label_num = len(train_dataset.targets.unique())


class CustomNetwork:
    def __init__(self, param_train_images, param_train_labels, param_label_num, lr=1e-6):
        """

        :param param_train_images: should be a transposed mathematics matrix
        :param param_train_labels: should be a transposed one hot mathematics matrix
        :param param_label_num:
        """

        self.train_images_t = param_train_images
        self.train_labels_t = param_train_labels
        self.label_num = param_label_num
        self.layer_num = 4
        self.lr = lr
        node_nums = (train_images.shape[-1], 30, 20, param_label_num)
        self.weights = []
        self.biases = []
        for i in range(1, len(node_nums)):
            weight = torch.randn(node_nums[i], node_nums[i - 1]) * (1 / node_nums[i - 1] ** 0.5)
            bias = torch.randn(node_nums[i]) * (1 / node_nums[i] ** 0.5)
            self.weights.append(weight)
            self.biases.append(bias)

        self.func = torch.sigmoid
        self.d_func = lambda y: y * (1 - y)  # torch.tensor -> torch.tensor
        self.residuals_t = [torch.empty(0) for _ in range(label_num)]
        self.activations_t = [torch.empty(0) for _ in range(label_num)]
        self.deltas_t = [torch.empty(0) for _ in range(label_num)]
        self.grads = [torch.empty(0) for _ in range(label_num - 1)]
        self.grad_biases = [torch.empty(0) for _ in range(label_num - 1)]

    def forward(self, data_t: torch.Tensor):
        self.activations_t[0] = data_t
        batch_size = data_t.shape[0]
        for i in range(0, self.label_num - 1):
            self.residuals_t[i + 1] = self.activations_t[i] @ self.weights[i] \
                                      + torch.ones((batch_size, 1)) @ self.biases[i]
            self.activations_t[i + 1] = self.func(self.residuals_t[i + 1])

    def backward(self, labels_t: torch.Tensor):
        self.deltas_t[-1] = -1 * (labels_t - self.activations_t[-1]) * self.d_func(self.residuals_t[-1])
        for i in range(self.label_num - 2, 0, -1):
            self.deltas_t[i] = (self.deltas_t[i + 1] @ self.weights[i]) * self.d_func(self.residuals_t[i])
        for i in range(0, self.label_num - 1):
            self.grads[i] = torch.einsum('ni,nj->nij', self.activations_t[i], self.deltas_t[i + 1])

    def update_params(self):
        pass

    def train(self):
        pass
