For Audio-MAE

1. MW-MAE — Multi-Window Local-Global Attention (ICLR 2024)

Paper: "Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners" (Yadav et al.)

MW-MAE proposes a Multi-Window Multi-Head Attention (MW-MHA) module that models local-global interactions in every decoder transformer block through attention heads of several distinct local and global windows. The key insight is that vanilla AudioMAE uses a single window size for all decoder heads, so every head captures the same scale of context. MW-MAE assigns each head a different window size (e.g. 2, 4, 8, global), so the decoder simultaneously sees fine phonemic detail and broad utterance-level structure in every block.

MW-MAE consistently outperforms standard MAEs across 10 downstream audio tasks, achieving the highest overall normalized score of 92.6±0.2, and it shows better performance in low-data scenarios — important for your constrained GSCv2 setup.

How to apply: Swap the decoder's standard multi-head attention for MW-MHA. It's a drop-in replacement with no change to the encoder or training objective. This directly combats JEPA's advantage of capturing richer semantics by making the decoder smarter.



2. AudioMAE++ — Macaron Blocks + SwiGLU (2025)

Paper: "AudioMAE++: Learning Better Masked Audio Representations with SwiGLU FFNs" (Yadav et al.)

AudioMAE++ proposes two architectural enhancements over vanilla AudioMAE: macaron-style transformer blocks and gated linear units (SwiGLU FFNs). When pretrained on AudioSet, it outperforms existing MAE-based approaches on 10 diverse downstream tasks and demonstrates excellent scaling characteristics, outperforming comparable standard MAE baselines with up to 4× more parameters. arXiv

How to apply: Replace the standard MLP block in your TransformerBlock with a macaron-style block (pre-norm → attention → pre-norm → FFN → pre-norm → FFN) and swap GELU for SwiGLU (x * sigmoid(gate)). This is a ~20-line change to backbone.py and has no impact on the masking or training pipeline.



4. data2vec Target (Multi-Layer Feature Prediction)

Paper: "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language" (Baevski et al. 2022)

Instead of reconstructing raw spectrogram pixels, the MAE decoder predicts the average of the top-K teacher encoder layer representations. This means the reconstruction target is already a semantic, contextualised feature — much closer to what JEPA predicts. EAT explicitly adopts data2vec 2.0's approach of regressing representations across multiple neural network layers, rather than concentrating exclusively on the top layer.

How to apply: Add a frozen EMA copy of your encoder. During training, for each masked patch, the target is the average of the last K=4 transformer block outputs from the EMA encoder (not the raw mel value). This is the single most impactful change you can make to vanilla MAE to close the gap with JEPA.



For Audio-JEPA

5. A-JEPA — Curriculum Masking (arxiv 2311.15830 — your reference paper)

Paper: "A-JEPA: Joint-Embedding Predictive Architecture Can Listen" (Fei et al.)

A-JEPA's masking strategy is designed in a curriculum manner for the audio spectrogram — gradually transitioning from random block masking to time-frequency aware masking on a schedule. This is important because random block masking in audio is "too easy" due to strong correlations along both time and frequency axes, so harder structured masking is needed later in training to keep the learning signal meaningful.

How to apply: Add a masking scheduler to masking.py that starts with block_masking() and transitions to jepa_masking() with structured time-frequency blocks after N warm-up epochs. A simple linear curriculum over the first 30% of training epochs works well.



6. Audio-JEPA (2025) — Clean Baseline

Paper: "Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning" (Tuncay et al., arxiv 2507.02915)

Audio-JEPA uses a simple ViT backbone to predict latent representations of masked spectrogram patches, pre-trained with random patch masking on mel-spectrograms. Although it is a straightforward translation of I-JEPA to audio, it shows comparable performance to wav2vec 2.0 and data2vec while using less than one-fifth of their training data with no hyperparameter tuning. arXiv

This is the most direct implementation of what you're building as the JEPA branch, and it confirms that even a clean implementation is very data-efficient — which matters for your GSCv2 constrained setting.