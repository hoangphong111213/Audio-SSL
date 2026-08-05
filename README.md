# Audio Self-Supervised Learning: Audio-MAE vs Audio-JEPA

This repository contains the implementation of two self-supervised learning paradigms for audio representation learning:

- **Audio-MAE** (Masked Autoencoder)
- **Audio-JEPA** (Joint Embedding Predictive Architecture)

Both methods are implemented from scratch using Vision Transformer backbones and evaluated under a unified experimental pipeline.

Unlike computer vision, where predictive objectives such as JEPA have shown strong advantages, our experiments demonstrate that **reconstruction-based learning consistently produces more discriminative representations for spectrogram-based audio under limited-data settings**.

The project was completed as the final course project for **Self-supervised Deep Learning**.
