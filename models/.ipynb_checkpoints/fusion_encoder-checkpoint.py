import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    """
    Multimodal Fusion Encoder that combines visual, behavioral, and textual embeddings.
    Uses a Transformer encoder to model cross-modal relationships and outputs usability scores.
    """
    def __init__(self):
        super(MultimodalFusion, self).__init__()
        self.fusion_proj = nn.Linear(512 * 3, 2048)  # Fuse three branches
        encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(2048, 5)  # Predict 5 usability metrics

    def forward(self, vision_x, behavioral_x, textual_x):
        """
        Args:
            vision_x (torch.Tensor): Visual features (batch_size, 512)
            behavioral_x (torch.Tensor): Behavioral features (batch_size, 512)
            textual_x (torch.Tensor): Textual features (batch_size, 512)

        Returns:
            torch.Tensor: Usability scores (batch_size, 5)
        """
        x = torch.cat([vision_x, behavioral_x, textual_x], dim=-1)
        x = self.fusion_proj(x).unsqueeze(1)  # (batch_size, 1, 2048)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Mean pooling
        x = self.fc(x)
        return x