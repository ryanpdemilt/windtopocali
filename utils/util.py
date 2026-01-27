import torch

def compute_metrics(target,pred):
    mse = torch.sum(torch.sqrt((target - pred)**2))
    return mse

def log_results(metrics):
    pass
