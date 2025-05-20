import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Replace the loss function with a custom loss function
# that computes the quantile loss for each quantile
class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(MultiQuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred_list, y_true):
        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            y_pred = y_pred_list[i].squeeze(-1)
            error = y_true - y_pred
            q_loss = torch.max(q * error, (q - 1) * error)
            total_loss += torch.mean(q_loss)
        return total_loss / len(self.quantiles)


class MCDropout(nn.Dropout):
    """MC Dropout stays active at inference."""
    def forward(self, input):
        return F.dropout(input, self.p, training=True, inplace=self.inplace)


class MQNN(torch.nn.Module):
    def __init__(self, input_dim, quantiles, internal_nodes=[32, 32], montecarlo=True):
        super(MQNN, self).__init__()
        self.quantiles = quantiles
        self.montecarlo = montecarlo

        layers = []
        prev_dim = input_dim

        for i, n_nodes in enumerate(internal_nodes):
            layers.append(nn.Linear(prev_dim, n_nodes))
            layers.append(nn.ReLU())
            if i != len(internal_nodes) - 1:
                layers.append(MCDropout(p=0.1) if montecarlo else nn.Dropout(p=0.1))
            prev_dim = n_nodes

        self.backbone = nn.Sequential(*layers)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(prev_dim, 1) for _ in quantiles
        ])

    def forward(self, x):
        features = self.backbone(x)
        outputs = [head(features) for head in self.quantile_heads]
        return outputs

def train_mqnn_model(train_loader, quantiles, internal_nodes=[128, 128],
                     montecarlo=True, epochs=100, batch_size=32, lr=0.001, device='cuda', verbose=1):

    input_dim = x_train.shape[1]
    model = MQNN(input_dim=input_dim, quantiles=quantiles,
                 internal_nodes=internal_nodes, montecarlo=montecarlo).to(device)

    criterion = MultiQuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred_list = model(x_batch)
            loss = criterion(y_pred_list, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        if verbose:
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model