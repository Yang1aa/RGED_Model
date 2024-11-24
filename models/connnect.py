from torch import nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim=2, dropout_rate=0.8):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits