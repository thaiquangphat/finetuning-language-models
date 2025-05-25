# Training Setup

This document provides a step-by-step guide for fine-tuning large language models. For implementation details, refer to `train.ipynb`.

## Model, Dataset & Fine-tuning Method
To configure the fine-tuning setup, select the appropriate indices for model, dataset, and fine-tuning method from the following options:

```python
# List of implemented methods
models     = ['t5-base', 'bart-base', 'flan-t5-small']
datasets   = ['squad', 'wmt16_en_de', 'imdb']
finetunes  = ['full', 'lora', 'adapters']

# Selecting index
model_idx, dataset_idx, finetune_idx = 1, 0, 2  # Corresponds to 'bart-base', 'squad', 'adapters'
```

## Finetuning Configuration

Customize the following hyperparameters in the `Hyperparameters` section of the notebook:

| Parameter            | Description                                          | Suggested Values         |
|----------------------|------------------------------------------------------|--------------------------|
| `num_train_epochs`   | Number of training epochs                            | 10–20                    |
|`learning_rate`| The learning rate for finetuning | `5e-4` for full-finetuning, `5e-4` for LoRA, and `1e-4` for Adapters. |
| `use_cpu`            | Whether to run training on CPU                       | `True` or `False`        |
| `train_batch_size`   | Batch size for training                              | 8–16                     |
| `eval_batch_size`    | Batch size for evaluation                            | 8–16                     |
| `test`               | If `True`, uses a small subset (20 samples) to test pipeline | `True` or `False`        |


## How To Run
Once you've set your parameters and configurations:

1. Open `train.ipynb`.
2. Modify the model, dataset, and method indices.
3. Adjust hyperparameters as needed.
4. Run all cells from top to bottom.

The fine-tuned model will be saved to:

```
models/ft-<model_name>-<method>-<task>
```

**Example:**
```
models/ft-bart-base-adapters-squad
```