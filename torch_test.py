import torch

if __name__ == "__main__":
    x = torch.Tensor([1,3,2])
    std,mean = torch.std_mean(x)
    torch.