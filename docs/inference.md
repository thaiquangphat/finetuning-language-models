# Inference Setup

This document provides a step-by-step guide for inference large language models. For implementation details, refer to `inference.ipynb`.

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

## Loading Dataset
We already handle to retrieve datasets. You are encouraged to only change the only following value.
- `test`: whether to inference on the full dataset (`False`), or take a sample of 20 samples (`True`). This is to ensure the model runs without errors.

## How To Run
Once you've set your parameters and configurations:

1. Open `inference.ipynb`.
2. Modify the model, dataset, and method indices.
3. Adjust `test` value.
4. Run all cells from top to bottom.

The inference results will be saved to:

```
prediction/prediction-<model_name>-<method>-<task>.json
```

**Example:**
```
prediction/prediction-bart-base-adapters-squad
```