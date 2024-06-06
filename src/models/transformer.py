import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerWithSalt(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, salt_layer=None):
        super(TransformerWithSalt, self).__init__()
        self.salt_layer = salt_layer
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        
    def forward(self, x):
        if self.salt_layer == 'input':
            x = x + torch.randn_like(x)  # Adding salt to the input
        
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        
        if self.salt_layer == 'output':
            x = x + torch.randn_like(x)  # Adding salt to the output
        
        x = self.fc(x[:, -1, :])
        return x
