
"""Training loop skeleton for STGNN."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..models.stgnn import STGNN
from .loss_functions import heteroscedastic_gaussian_nll

def train_dummy(adj, x_seq, y, epochs=5, lr=1e-3):
    """Minimal training example on in-memory tensors."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGNN(in_dim=x_seq.shape[-1], hidden_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(
        torch.from_numpy(x_seq).float(),
        torch.from_numpy(y).float()
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    adj_t = torch.from_numpy(adj).float().to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            # Here we ignore batch dimension for simplicity
            batch_x = batch_x[0].to(device)  # (T, N, F)
            batch_y = batch_y.to(device)     # (N, 1)
            optimizer.zero_grad()
            mean, logvar = model(batch_x, adj_t)
            loss = heteroscedastic_gaussian_nll(batch_y, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")
