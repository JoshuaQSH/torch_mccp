import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MCDropout(nn.Dropout):
    """Dropout that remains active during inference for MC Dropout."""
    def forward(self, input):
        return F.dropout(input, self.p, training=True, inplace=self.inplace)

# input -> Conv2d (32, 3, 3) -> ReLU -> MaxPool2d (2, 2) -> Dropout (0.5) -> Conv2d (64, 3, 3) -> ReLU -> MaxPool2d (2, 2) -> Dropout (0.5) -> Flatten -> Linear (128) -> ReLU -> Linear (output_dims) -> Softmax
class CNN_MC(nn.Module):
    def __init__(self, input_shape, output_dims, montecarlo=True, activation=F.relu):
        super(CNN_MC, self).__init__()
        self.montecarlo = montecarlo
        self.activation = activation

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = MCDropout(p=0.5) if montecarlo else nn.Dropout(p=0.5)
        # self.drop1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = MCDropout(p=0.5) if montecarlo else nn.Dropout(p=0.5)
        # self.drop2 = nn.Dropout(p=0.5)

        self.flatten_dim = self._get_flattened_size(input_shape)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, output_dims)

    def _get_flattened_size(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            x = self.pool1(self.activation(self.conv1(dummy)))
            x = self.pool2(self.activation(self.conv2(self.drop1(x))))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        # Conv2d (32, 3, 3) -> ReLU -> MaxPool2d (2, 2) 
        x = self.pool1(self.activation(self.conv1(x)))
        # Dropout (0.5)
        x = self.drop1(x)
        # Conv2d (64, 3, 3) -> ReLU -> MaxPool2d (2, 2)
        x = self.pool2(self.activation(self.conv2(x)))
        # Dropout (0.5)
        x = self.drop2(x)
        
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
