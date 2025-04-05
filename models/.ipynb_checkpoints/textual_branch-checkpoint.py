import torch.nn as nn
from transformers import RobertaModel

class TextualBranch(nn.Module):
    """
    Textual Branch that processes user feedback text using a RoBERTa encoder.
    Captures semantic information from textual input.
    """
    def __init__(self):
        super(TextualBranch, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.projection = nn.Linear(768, 512)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): Tokenized input text (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask for input text (batch_size, sequence_length)

        Returns:
            torch.Tensor: Encoded textual features (batch_size, 512)
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        x = self.projection(cls_output)
        return x