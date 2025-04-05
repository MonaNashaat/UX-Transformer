# Usability Evaluation Framework

This project implements a deep multimodal framework for evaluating website usability through visual, behavioral, and textual inputs.

## Project Structure

usability_framework/
<br/>├── models/
<br/>│   ├── vision_branch.py
<br/>│   ├── behavioral_branch.py
<br/>│   ├── textual_branch.py
<br/>│   ├── fusion_encoder.py
<br/>│   └── usability_framework.py
<br/>├── train.py
<br/>├── utils.py
<br/>├── config.py
<br/>├── README.md

## Architecture Overview

- Visual Branch: Processes UI screenshots via Vision Transformer (ViT)
- Behavioral Branch: Models user interaction sequences via Temporal Transformer
- Textual Branch: Processes feedback using RoBERTa
- Multimodal Fusion Encoder: Combines modalities and outputs usability scores

## Training

1. Prepare your dataset (screenshots, behavior logs, feedback).
2. Update `utils.py` with your data loading logic.
3. Train the model:

```bash
python train.py
```
Requirements
	•	torch
	•	torchvision
	•	transformers

Install dependencies:
```bash
pip install torch torchvision transformers
```
