import torch

A = torch.tensor([[1, 1, 1,1], [2, 2, 2,2], [3, 3, 3,3]])
B = torch.tensor([1, 2, 3,4])

print(torch.nn.functional.one_hot(B))