import torch
from torch.utils.data import DataLoader, random_split

def split_test_set(dataset, n_cal=2500, batch_size=64, shuffle=False):
    n_total = len(dataset)
    n_val = n_total - n_cal
    cal_dataset, val_dataset = random_split(dataset, [n_cal, n_val])
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return cal_loader, val_loader