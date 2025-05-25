# Model Module

This directory includes model definitions and configurations for fine-tuning language models. It contains:

- `models.py`: Script defining the model architecture and utilities.
- `llms/`: Subdirectory with implementations for specific language models:
  - `bart.py`: Implementation for BART.
  - `t5.py`: Implementation for T5.

## Model Configuration

### Loading Models
```python
from modules.model.models import load_t5_base, load_bart_base

# Load T5 model
model, tokenizer = load_t5_base(
    name="t5-base",
    finetune_type="lora",
    task="question_answering",
    device="cuda"
)

# Load BART model
model, tokenizer = load_bart_base(
    name="bart-base",
    finetune_type="full",
    task="translation",
    device="cuda"
)
```

### Fine-tuning Types
1. **Full Fine-tuning**
   - Updates all model parameters
   - Highest memory requirements
   - Best performance but slowest training

2. **LoRA (Low-Rank Adaptation)**
   - Adds trainable rank decomposition matrices
   - Memory efficient
   - Good performance with faster training

3. **Adapters**
   - Adds small trainable modules between layers
   - Most memory efficient
   - Slightly lower performance but fastest training

### Task-Specific Configurations

#### Question Answering
- Input format: 
  ```python
  {
      "input": "question: {question} context: {context}",
      "target": "{answer}"
  }
  ```
- Uses SQuAD dataset format

#### Text Sentiment Analysis
- Input format:
  ```python
  {
      "input": "sentiment analysis: {text}",
      "target": "sentiment: {True/False}"  # True for positive, False for negative
  }
  ```
- Binary classification task
- Uses IMDb dataset format

#### Translation
- Input format:
  ```python
  {
      "input": "translate to german. english: {english_text}",
      "target": "german: {german_text}"
  }
  ```
- Supports English to German translation
- Uses WMT16 dataset format

## Model Architecture Details

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