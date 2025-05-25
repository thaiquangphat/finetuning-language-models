# Finetuning Language Models

This repository provides tools and scripts for fine-tuning language models on various tasks, such as sentiment analysis, question answering, and text translation. It includes preprocessing utilities, training pipelines, and inference modules to streamline the fine-tuning process.

## Features

- **Data Preprocessing**: Utilities for preparing datasets, including tokenization and formatting for models like Hugging Face Transformers.
- **Training**: Scripts for training language models with custom configurations.
- **Evaluation**: Tools for evaluating model performance on test datasets.
- **Prediction**: Modules for running inference on new data.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Training Resources

#### Hardware Requirements
- **Recommended GPU**: NVIDIA T4 or better
  - T4 GPU provides 16GB VRAM, suitable for fine-tuning medium-sized models
  - Supports mixed precision training (FP16)
  - Good balance of performance and cost

#### Cloud Platforms
- **[Kaggle](https://www.kaggle.com/)**
  - Free T4 GPU access (30 hours/week)
  - Easy-to-use notebook interface
  - Built-in dataset integration
  - Steps to use:
    1. Create a Kaggle account
    2. Create a new notebook
    3. Enable GPU accelerator in notebook settings
    4. Clone this repository
    5. Install requirements
    6. Start training

#### Alternative Platforms
- **Google Colab**
  - Free T4 GPU access (limited hours)
  - Similar notebook interface to Kaggle
- **AWS SageMaker**
  - Pay-as-you-go GPU instances
  - More flexible but requires AWS account
- **Google Cloud Platform**
  - Various GPU options
  - Requires GCP account and billing setup

#### Memory Management Tips
- Use gradient accumulation for larger batch sizes
- Enable mixed precision training (FP16)
- Monitor GPU memory usage during training
- Adjust batch size based on model size and available VRAM

### Project Structure

```
finetuning-language-models/
├── modules/                      # Core functionality modules
│   ├── data/                    # Data handling and preprocessing
│   │   ├── hfdata/             # HuggingFace dataset implementations
│   │   ├── hfdatasets/         # Dataset loading and preprocessing functions
│   │   └── config.json         # Dataset configuration file
│   │
│   ├── evaluate/               # Evaluation metrics and utilities
│   │   ├── question_answering/ # QA-specific evaluation files
│   │   ├── text_sentiment/     # Sentiment analysis evaluation files
│   │   ├── translation/        # Translation evaluation files
│   │   ├── eval_qa.py         # Question answering evaluation
│   │   ├── eval_text_sentiment.py # Sentiment analysis evaluation
│   │   ├── eval_translation.py # Translation evaluation
│   │   └── ultis.py           # Evaluation utilities
│   │
│   ├── inference/             # Model inference and prediction
│   │   ├── inference.py       # Core inference functionality
│   │   └── README.md         # Inference documentation
│   │
│   ├── model/                # Model implementations
│   │   ├── llms/            # Language model implementations
│   │   │   ├── t5.py       # T5 model implementation
│   │   │   ├── bart.py     # BART model implementation
│   │   │   └── gpt2.py     # GPT-2 model implementation
│   │   ├── models.py        # Model loading and configuration
│   │   └── README.md        # Model documentation
│   │
│   └── train/               # Training utilities
│       ├── trainer.py       # Training pipeline implementation
│       ├── ultis.py         # Training utilities
│       └── README.md        # Training documentation
│
├── docs/                    # Project documentation
│   ├── train.md            # Training instructions
│   ├── eval.md             # Evaluation instructions
│   └── results.md          # Results and analysis
│
├── train.ipynb             # Training notebook
├── inference.ipynb         # Inference notebook
├── requirements.txt        # Project dependencies
├── api_key.json           # API key configuration
├── LICENSE                # Project license
└── README.md             # Project documentation
```

#### Directory Explanations

- **modules/**: Core functionality directory containing all implementation modules
  - **data/**: Handles dataset loading, preprocessing, and configuration
  - **evaluate/**: Contains evaluation metrics and utilities for different tasks
  - **inference/**: Manages model inference and prediction functionality
  - **model/**: Implements different language models and their configurations
  - **train/**: Contains training pipeline and related utilities

- **docs/**: Project documentation and instructions
  - Contains detailed guides for training, evaluation, and results analysis

- **Root Files**:
  - `train.ipynb`: Interactive notebook for model training
  - `inference.ipynb`: Interactive notebook for model inference
  - `requirements.txt`: Project dependencies
  - `api_key.json`: API key configuration for external services
  - `LICENSE`: Project license information

### Documents

Please refer to the following documents for detailed instructions and results:

| File                                | Purpose                  |
|-------------------------------------|--------------------------|
| [docs/train.md](docs/train.md)     | Fine-tuning instructions |
| [docs/eval.md](docs/eval.md)       | Inference instructions  |
| [docs/results.md](docs/results.md) | Evaluation instructions and result       |

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the repository.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for providing state-of-the-art NLP tools.
- The following datasets:
    - SQuAD dataset for question answering tasks.
    - IMDb dataset for sentiment analysis tasks.
    - WMT dataset for translation from English to German tasks.
