# Inference Module

This directory contains the inference implementation for fine-tuned language models. The module provides a unified interface for running predictions across different tasks and model architectures.

## Components

- `inference.py`: Core inference script that handles:
  - Model loading and configuration
  - Input preprocessing
  - Batch prediction
  - Output post-processing
  - Support for different model architectures (T5, BART)

## Usage

```python
from modules.inference.inference import run_inference

# Run inference on a test dataset
results = run_inference(
    model_name="t5-base",
    task="question_answering",
    test_data=test_dataset,
    device="cuda"
)
```

## Features

- **Multi-task Support**: Handles various NLP tasks:
  - Question Answering
  - Text Sentiment Analysis
  - Machine Translation

- **Model Compatibility**: Works with supported model architectures:
  - T5-based models (T5-Base, Flan-T5-Small)
  - BART-based models (BART-Base)

- **Flexible Input**: Accepts both single examples and batched inputs

- **Output Formatting**: Automatically formats outputs according to task requirements

## Best Practices

1. **Memory Management**:
   - Use appropriate batch sizes based on model size
   - Enable mixed precision (FP16) for faster inference
   - Clear GPU cache between large inference runs

2. **Performance Optimization**:
   - Use GPU acceleration when available
   - Implement batch processing for multiple inputs
   - Cache model and tokenizer for repeated inference

3. **Error Handling**:
   - Graceful handling of out-of-memory situations
   - Input validation and sanitization
   - Proper error messages for debugging

## Examples

### Question Answering
```python
# Example input format
input_text = {
    "input": "question: What does the fox do? context: The quick brown fox jumps over the lazy dog.",
    "target": "jumps over the lazy dog"
}

# Run inference
answer = run_inference(model, input_text, task="question_answering")
```

### Text Sentiment Analysis
```python
# Example input format
input_text = {
    "input": "sentiment analysis: I really enjoyed this movie, it was fantastic!",
    "target": "sentiment: True"  # True for positive, False for negative
}

# Run inference
sentiment = run_inference(model, input_text, task="text_sentiment")
```

### Translation
```python
# Example input format
input_text = {
    "input": "translate to german. english: Hello, how are you?",
    "target": "german: Hallo, wie geht es dir?"
}

# Run inference
translation = run_inference(model, input_text, task="translation")
```