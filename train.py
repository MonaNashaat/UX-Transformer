import torch
from torch.utils.data import DataLoader
from models.usability_framework import UsabilityFramework
from utils import load_data, compute_loss
from config import device, learning_rate, num_epochs

# Initialize model
model = UsabilityFramework().to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Load datasets
train_loader, val_loader = load_data()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        images, behaviors, input_ids, attention_masks, labels = batch
        images, behaviors, input_ids, attention_masks, labels = images.to(device), behaviors.to(device), input_ids.to(device), attention_masks.to(device), labels.to(device)

        outputs = model(images, behaviors, input_ids, attention_masks)
        loss = compute_loss(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")

print("Training complete.")