import torch.nn as nn

class BehavioralBranch(nn.Module):
    """
    Behavioral Branch that processes user interaction sequences using a Transformer encoder.
    Models temporal dependencies between user actions.
    """
    def __init__(self):
        super(BehavioralBranch, self).__init__()
        self.embedding = nn.Linear(10, 512)  # Event feature dimension = 10
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of interaction sequences (batch_size, sequence_length, feature_dim)

        Returns:
            torch.Tensor: Encoded behavioral features (batch_size, 512)
        """
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Mean pooling over sequence length
        return x