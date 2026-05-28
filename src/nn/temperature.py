import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T=1

    def forward(self, logits):
        return logits / torch.exp(self.log_T)

def learn_temperature(model, val_loader):
    model.eval()
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=0.001, max_iter=50)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in val_loader:
            z = model(x)
            logits_list.append(z)
            labels_list.append(y)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    def closure():
        optimizer.zero_grad()
        z_scaled = scaler(logits)
        loss = F.binary_cross_entropy_with_logits(z_scaled, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)

    return torch.exp(scaler.log_T).item()