import torch
import torch.nn as nn

class TemporalMemoryBank(nn.Module):
    """
    Temporal Memory Bank (TMB)
    Processes a sequence of frame embeddings to predict a video-level scalar (EF).
    """
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, hidden_dim)) # Max seq len 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # EF is usually 0-1
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            ef: (B, 1)
        """
        B, T, C = x.shape
        
        # Project and add pos encoding
        x = self.input_proj(x) # (B, T, hidden_dim)
        x = x + self.pos_encoder[:, :T, :]
        
        # Encode
        x = self.transformer(x) # (B, T, hidden_dim)
        
        # Global Average Pooling over time
        x = x.mean(dim=1) # (B, hidden_dim)
        
        # Predict
        ef = self.head(x)
        return ef
