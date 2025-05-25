# Training Module

This directory contains the training implementation for fine-tuning language models. The module provides a comprehensive training pipeline with support for various fine-tuning techniques and optimization strategies.

## Components

- `trainer.py`: Main training script that implements:
  - Model initialization and configuration
  - Training loop management
  - Weights & Biases integration
  - Checkpoint saving and loading
  - Custom training behaviors

- `ultis.py`: Utility functions including:
  - `ExtractiveQATrainer`: Custom trainer for extractive QA tasks
  - `LeastTrainLossTrainer`: Trainer that saves models with lowest training loss
  - Debug printing utilities

## Features

### Training Pipeline
- **Flexible Fine-tuning**: Support for multiple fine-tuning approaches:
  - Full fine-tuning
  - LoRA (Low-Rank Adaptation)
  - Adapter-based fine-tuning

- **Optimization Features**:
  - Mixed precision training (FP16)
  - Gradient accumulation
  - Learning rate scheduling
  - Weight decay
  - Early stopping

- **Monitoring and Logging**:
  - Weights & Biases integration
  - Training metrics tracking
  - Model checkpointing
  - Custom evaluation metrics

## Usage

```python
from modules.train.trainer import BaseTrainer

# Initialize trainer
trainer = BaseTrainer(
    device="cuda",
    model="t5-base",
    dataset="squad",
    finetune="lora",
    train_batch_size=8,
    eval_batch_size=16
)

# Configure W&B
trainer.set_wandb_api(
    wandb_token="your_token",
    wandb_api="your_api_key",
    project="your_project"
)

# Start training
trainer.run(
    saved_model="path/to/save",
    num_train_epochs=10,
    learning_rate=3e-4,
    weight_decay=0.02
)
```

## Best Practices

1. **Resource Management**:
   - Monitor GPU memory usage
   - Use appropriate batch sizes
   - Enable gradient accumulation for larger effective batch sizes

2. **Training Stability**:
   - Use learning rate warmup
   - Implement gradient clipping
   - Monitor loss curves
   - Save checkpoints regularly

3. **Performance Optimization**:
   - Enable mixed precision training
   - Use efficient data loading
   - Implement proper batching
   - Cache preprocessed data when possible

## Monitoring

Training progress can be monitored through:
- Weights & Biases dashboard
- Training logs
- Model checkpoints
- Evaluation metrics