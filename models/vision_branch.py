import torch.nn as nn
from torchvision.models import vit_b_16

class VisionBranch(nn.Module):
    """
    Vision Branch that processes UI screenshots using a Vision Transformer (ViT).
    Extracts spatial features and projects them into a 512-dimensional space.
    """
    def __init__(self):
        super(VisionBranch, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove the classification head
        self.projection = nn.Linear(768, 512)  # Project ViT output to 512 dimensions

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of images of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Encoded visual features (batch_size, 512)
        """
        x = self.vit(x)
        x = self.projection(x)
        return x