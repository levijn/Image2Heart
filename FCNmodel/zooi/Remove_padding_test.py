import torch

x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
y = torch.narrow(x, 2, 0, 2)

print(y)