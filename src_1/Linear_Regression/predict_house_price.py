"""
EXERCISE 1A
"""

import numpy as np
from matplotlib import pyplot as plt


def load_data(file_path="../../data/housing.data") -> np.ndarray:
    with open(file_path) as f:
        content = f.readlines()
        dataset = []
        for line in content:
            if line:
                data_list = line.split(" ")
                data_list = [num for num in data_list if num]
                data_list = list(map(float, data_list))
                if data_list:
                    dataset.append(data_list)
    return np.array(dataset)


def main():
    dataset = load_data()
    np.random.shuffle(dataset)
    mean = np.mean(dataset[:, :-1], axis=0)
    std = np.std(dataset[:, :-1], axis=0)
    dataset[:, :-1] = (dataset[:, :-1] - mean) / std
    # add one 1 column to express bias.
    bias_column = np.ones(dataset.shape[0]).reshape(-1, 1)
    dataset = np.hstack((bias_column, dataset))

    data_num = len(dataset)
    sample_rate = 0.8
    train_dataset = dataset[:int(data_num * sample_rate)]
    test_dataset = dataset[int(data_num * sample_rate):]

    feature_num = dataset.shape[-1] - 1  # number of features
    thetas = np.random.uniform(-0.001, 0.001, feature_num)  # 初始化在较小范围

    # random initialization in [0, 1)

    def get_loss(matrix, theta: np.ndarray):
        y_hats = np.dot(matrix[:, :-1], theta)
        loss_value = np.mean((y_hats - matrix[:, -1]) ** 2)
        return loss_value

    learn_rate = 0.0001

    def nabla(matrix, theta):
        bias: np.ndarray = np.dot(matrix[:, :-1], theta) - matrix[:, -1]
        res = np.dot(bias, matrix[:, :-1])

        return res

    epoch_num = 400
    losses = []  # List to store loss values

    for e in range(epoch_num):
        thetas = thetas - learn_rate * nabla(train_dataset, thetas)
        loss = get_loss(train_dataset, thetas)
        losses.append(loss)  # Record loss value
        print(f"Epoch {e + 1}, Loss: {loss}, Thetas: {thetas}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(epoch_num), losses, label="Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.show()

    # Evaluation
    y_hats = np.dot(test_dataset[:, :-1], thetas)
    y_labels = test_dataset[:, -1]

    # Sort by true labels
    sorted_indices = np.argsort(y_labels)
    y_labels_sorted = y_labels[sorted_indices]
    y_hats_sorted = y_hats[sorted_indices]

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_labels_sorted)), y_labels_sorted, label="True Labels", alpha=0.7)
    plt.scatter(range(len(y_hats_sorted)), y_hats_sorted, label="Predicted Values", alpha=0.7)
    plt.xlabel("Sample Index (Sorted by True Values)")
    plt.ylabel("Value")
    plt.title("True Labels vs. Predicted Values")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
