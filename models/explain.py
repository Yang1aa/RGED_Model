from torch import nn

class ExplanationLayer(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(ExplanationLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, evidence_features):
        attn_output, attn_weights = self.attention(evidence_features, evidence_features, evidence_features)
        explanation = attn_output.mean(dim=1)
        explanation = self.relu(self.fc(explanation))
        return explanation
