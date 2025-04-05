import torch.nn.functional as F

def load_data():
    """
    Placeholder for dataset loading.
    Replace with your actual DataLoader setup.
    Returns:
        train_loader (DataLoader), val_loader (DataLoader)
    """
    return None, None  # Replace with real DataLoader

def compute_loss(outputs, labels):
    """
    Computes Mean Squared Error loss between predicted and target usability scores.

    Args:
        outputs (torch.Tensor): Model predictions (batch_size, 5)
        labels (torch.Tensor): True scores (batch_size, 5)

    Returns:
        torch.Tensor: Loss value
    """
    return F.mse_loss(outputs, labels)