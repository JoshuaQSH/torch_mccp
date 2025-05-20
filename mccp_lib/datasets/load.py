import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_tensor_transform(image_shape):
    return transforms.Compose([
        transforms.Resize(image_shape[:2]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def load_cifar10(batch_size=128, shuffle=True, num_workers=2):
    transform = get_tensor_transform((32, 32))
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def load_cifar100(batch_size=128, shuffle=True, num_workers=2):
    transform = get_tensor_transform((32, 32))
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def load_mnist(up_scale=False, channel_first=False, batch_size=128, shuffle=True, num_workers=2):
    if up_scale:
        transform_list = [
            transforms.Resize(32),  # Upscale from 28x28 to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    else:
        image_shape = (1, 28, 28)
        transform_list = [
            transforms.Resize(image_shape[:2]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

    if channel_first:
        # A 3 channels imaghe if needed
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  
    
    transform = transforms.Compose(transform_list)
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def load_fashion_mnist(up_scale=False, channel_first=False, batch_size=128, shuffle=True, num_workers=2):
    if up_scale:
        transform_list = [
            transforms.Resize(32),  # Upscale from 28x28 to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    else:
        image_shape = (1, 28, 28)
        transform_list = [
            transforms.Resize(image_shape[:2]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

    if channel_first:
        # A 3 channels imaghe if needed
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  
    
    transform = transforms.Compose(transform_list)
    
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def prepare_tabular_dataset(x, y, normalize=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    input_shape = x_train.shape[1]

    idx = np.random.permutation(len(x_train))
    mid = len(idx) // 2
    idx_train, idx_cal = idx[:mid], idx[mid:2*mid]

    if normalize:
        scaler = StandardScaler()
        scaler.fit(x_train[idx_train])
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    mean_y = np.mean(np.abs(y_train[idx_train]))
    y_train = y_train / mean_y
    y_test = y_test / mean_y

    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), input_shape


def load_concrete(path_to_dataset='./data/concrete_data.csv'):
    columns = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
               "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete compressive strength"]
    df = pd.read_csv(path_to_dataset, skiprows=1, names=columns)
    y = df['Concrete compressive strength'].values.astype(np.float32)
    x = df.drop('Concrete compressive strength', axis=1).values.astype(np.float32)
    return prepare_tabular_dataset(x, y)


def load_abalone(path_to_dataset='./data/abalone_data.csv'):
    columns = ["sex", "length", "diameter", "height", "whole weight", "shucked weight",
               "viscera weight", "shell weight", "rings"]
    df = pd.read_csv(path_to_dataset, skiprows=1, names=columns)
    for label in "MFI":
        df[label] = (df["sex"] == label).astype(float)
    df = df.drop(columns=["sex"])
    y = df['rings'].values.astype(np.float32)
    x = df.drop('rings', axis=1).values.astype(np.float32)
    return prepare_tabular_dataset(x, y)


def load_housing_prices():
    from keras.datasets import boston_housing
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return prepare_tabular_dataset(x, y)


def load_custom_dataset():
    raise NotImplementedError("Custom dataset loading not implemented.")