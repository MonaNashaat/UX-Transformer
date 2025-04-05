# Usability Evaluation Framework

This project implements a deep multimodal framework for evaluating website usability through visual, behavioral, and textual inputs.

## Project Structure
usability_framework/
├── models/
│   ├── vision_branch.py
│   ├── behavioral_branch.py
│   ├── textual_branch.py
│   ├── fusion_encoder.py
│   └── usability_framework.py
├── train.py
├── utils.py
├── config.py
├── README.md

## Architecture Overview

The following diagram illustrates the structure of the framework:

![Usability Framework Flowchart](A_flowchart_diagram_in_the_image_illustrates_a_usa.png)

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

pip install torch torchvision transformers
