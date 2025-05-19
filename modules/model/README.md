# Model Module

This directory includes model definitions and configurations for fine-tuning language models. It contains:

- `models.py`: Script defining the model architecture and utilities.
- `llms/`: Subdirectory with implementations for specific language models:
  - `bart.py`: Implementation for BART.
  - `gpt2.py`: Implementation for GPT-2.
  - `t5.py`: Implementation for T5.



# Overview of Selected Language Models

This repository includes experiments and fine-tuning workflows for the following transformer-based models:

## 1. T5-Base
- **Origin**: Developed by Google Research.
- **Type**: Text-to-Text Transformer (Encoder-Decoder).
- **Number of Parameters**: ~220 million.
- **Description**: T5 (Text-to-Text Transfer Transformer) formulates all NLP tasks as a text-to-text problem, providing a unified approach to sequence modeling. The base variant offers a balance between model size and performance, making it suitable for a wide range of downstream tasks.
- **Citation**:  
  Raffel, Colin, et al. "*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*." JMLR, 2020. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- **License**: Apache 2.0

## 2. BART-Base
- **Origin**: Developed by Facebook AI.
- **Type**: Sequence-to-Sequence Model (Encoder-Decoder).
- **Number of Parameters**: ~139 million.
- **Description**: BART (Bidirectional and Auto-Regressive Transformers) combines the benefits of bidirectional context encoding (like BERT) and autoregressive decoding (like GPT), making it highly effective for generative and comprehension tasks.
- **Citation**:  
  Lewis, Mike, et al. "*BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*." ACL, 2020. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)
- **License**: MIT License

## 3. Flan-T5-Small
- **Origin**: Developed by Google Research.
- **Type**: Instruction-Tuned Text-to-Text Transformer (Encoder-Decoder).
- **Number of Parameters**: ~80 million.
- **Description**: Flan-T5 is a fine-tuned version of T5 using instruction-based datasets, improving its generalization on unseen tasks with minimal supervision. The small variant is lightweight and efficient for prototyping and low-resource applications.
- **Citation**:  
  Chung, Hyung Won, et al. "*Scaling Instruction-Finetuned Language Models*." 2022. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416)
- **License**: Apache 2.0

---

## Notes
- **T5-Base** and **Flan-T5-Small** share the same underlying architecture but differ in fine-tuning objectives.
- All models are compatible with the Hugging Face 🤗 Transformers library.