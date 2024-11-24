from torch import nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_channels, out_channels, num_relations)
        self.rgcn2 = RGCNConv(out_channels, out_channels, num_relations)
        self.rgcn3 = RGCNConv(out_channels, out_channels, num_relations)
        self.res_linear = nn.Linear(in_channels, out_channels)
        self.drop = nn.Dropout(0.05)

    def forward(self, x, edge_index, edge_type):
        res = self.res_linear(x)
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = F.relu(x)
        x = res + x
        x = F.relu(self.rgcn2(x, edge_index, edge_type))
        x = F.relu(x)
        x = res + x
        x = F.relu(self.rgcn3(x, edge_index, edge_type))
        x = F.relu(x)
        x = res + x
        x = self.drop(x)
        return x