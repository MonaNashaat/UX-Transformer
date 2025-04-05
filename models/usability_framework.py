import torch.nn as nn
from models.vision_branch import VisionBranch
from models.behavioral_branch import BehavioralBranch
from models.textual_branch import TextualBranch
from models.fusion_encoder import MultimodalFusion

class UsabilityFramework(nn.Module):
    """
    End-to-end Usability Evaluation Framework.
    Combines visual, behavioral, and textual analysis through deep multimodal learning.
    """
    def __init__(self):
        super(UsabilityFramework, self).__init__()
        self.vision_branch = VisionBranch()
        self.behavioral_branch = BehavioralBranch()
        self.textual_branch = TextualBranch()
        self.fusion = MultimodalFusion()

    def forward(self, image, behavior, input_ids, attention_mask):
        """
        Args:
            image (torch.Tensor): UI screenshot batch (batch_size, 3, 224, 224)
            behavior (torch.Tensor): User interaction sequences (batch_size, sequence_length, 10)
            input_ids (torch.Tensor): Tokenized feedback text (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention masks (batch_size, sequence_length)

        Returns:
            torch.Tensor: Predicted usability scores (batch_size, 5)
        """
        vision_x = self.vision_branch(image)
        behavioral_x = self.behavioral_branch(behavior)
        textual_x = self.textual_branch(input_ids, attention_mask)
        output = self.fusion(vision_x, behavioral_x, textual_x)
        return output