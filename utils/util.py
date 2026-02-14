import torch

def compute_metrics(target,pred):
    mse = torch.sum(torch.sqrt((pred-target)**2))
    return mse

def log_results(metrics):
    pass
