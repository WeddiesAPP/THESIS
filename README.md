# Uncertainty-Aware Image Captioning with BLIP-2

This repository contains the code used for my MSc Artificial Intelligence thesis:  
**"Integrating and Calibrating Uncertainty Expressions in Image Captioning Systems"** (Utrecht University, 2025). **(https://studenttheses.uu.nl/handle/20.500.12932/49908)**

The project explores how to fine-tune **BLIP-2 (OPT-2.7B)** with **LoRA** and **8-bit quantization** to generate captions that include **uncertainty expressions**. The dataset consists of images annotated with modified captions (scene, action, rationale) and corresponding confidence scores. The aim is to train a model that not only describes images but also expresses calibrated levels of certainty.

---

## Project Overview
- **Model**: [Salesforce BLIP-2 OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b)  
- **Fine-tuning**: Parameter-efficient fine-tuning with [PEFT/LoRA](https://huggingface.co/docs/peft/index)  
- **Quantization**: 8-bit loading with `bitsandbytes` to reduce GPU memory footprint  
- **Dataset**: Custom Hugging Face dataset based on High-Level Dataset (https://huggingface.co/datasets/michelecafagna26/hl) with:
  - Original image
  - Modified captions (scene/action/rationale)
  - Confidence scores (1–5 Likert scale)

---

## Key Features
- **Custom Dataset Class**  
  Each sample expands into multiple prompt–caption pairs (scene, action, rationale), with labels masked to avoid penalizing the prompt.  
- **Weighted Sampling**  
  Confidence bins are balanced with `WeightedRandomSampler` to mitigate label imbalance.  
- **Training Loop**  
  Implements gradient clipping, loss logging, checkpoint saving, and CSV exports of batch/epoch losses.  
- **Frozen Vision Encoder + Q-Former**  
  Only the OPT language model is adapted via LoRA.  

---

This project investigates how large multimodal models can be calibrated to express uncertainty.
Two main approaches were explored:

Supervised Fine-tuning (LoRA) on captions with explicit uncertainty markers.

Reinforcement Learning (PPO) using a trained confidence classifier as reward model.

Evaluation included both semantic similarity and confidence alignment between model outputs and annotated captions.
