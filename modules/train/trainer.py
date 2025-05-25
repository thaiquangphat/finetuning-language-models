# For training
from transformers import (
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
)
from modules.train.ultis import ExtractiveQATrainer # For Extractive Question Answering training
from modules.train.ultis import debug_print # For debugging

# For login wandb
import os
from huggingface_hub import login

# For preprocessing
from modules.data.hfdata import (
    SquadDataset, SquadDatasetExtractive, SquadDatasetDecoder,
    WMTDatasetDecoder, WMTDataset,
    IMDBDatasetDecoder, IMDBDataset,
)
from modules.data.hfdatasets.squad import load_squad, preprocess_squad
from modules.data.hfdatasets.wmt import load_wmt, preprocess_wmt
from modules.data.hfdatasets.imdb import load_imdb, preprocess_imdb
from modules.model.models import load_t5_base, load_bart_base, load_gpt_2
import json

# Task mapper
data2task = {
    "squad": "question_answering",
    "imdb": "text_sentiment_analysis",
    "wmt16_en_de": "english_to_german_translation"
}

# ===============================================================================================
def get_model_tokenizer(model, finetune_type, task, device):
    """
    Loads and configures a model and its tokenizer based on the specified parameters.
    
    This function serves as a factory for loading different model types (T5, BART, GPT-2)
    with their corresponding tokenizers.
    
    Args:
        model (str): Name of the model to load
        finetune_type (str): Type of fine-tuning to use ('full', 'lora', 'adapters')
        task (str): The NLP task to configure the model for
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: The configured model for the specified task
            - tokenizer: The corresponding tokenizer
            
    Raises:
        NotImplementedError: If the specified model is not implemented
    """
    loaders = {
        't5-base': load_t5_base,
        'bart-base': load_bart_base,
        'gpt2': load_gpt_2,
        'mt5-small': load_t5_base, # same loader as t5-base
        'flan-t5-small': load_t5_base, # same loader as t5-base
    }

    for key in loaders:
        if key in model:
            return loaders[key](model, finetune_type, task, device)

    raise NotImplementedError(
        f"Model '{model}' not implemented."
        f"Available options: {list(loaders.keys())}"
    )
    
def get_dataset(dataset, tokenizer, model_name, test):
    """
    Loads and preprocesses a dataset for training.
    
    This function handles loading and preprocessing datasets for different models:
    - For non-GPT models: Uses dataset-specific classes
    - For GPT-2: Uses custom preprocessing functions
    
    Args:
        dataset (str): Name of the dataset to load
        tokenizer: The tokenizer to use for preprocessing
        model_name (str): Name of the model being used
        test (bool): Whether to use test mode (smaller dataset)
        
    Returns:
        tuple: (train_data, test_data, val_data) containing the processed datasets
        
    Raises:
        NotImplementedError: If the specified dataset is not implemented
    """
    # get config from file
    with open('modules/data/config.json', 'r', encoding='utf-8') as file:
        data_config = json.load(file)
    
    # generic loader
    if 'gpt' not in model_name:
        loaders = {
            'squad': (load_squad, SquadDataset),
            'wmt16_en_de': (load_wmt, WMTDataset),
            'imdb': (load_imdb, IMDBDataset),
        }
        load_fn, DatasetClass = loaders[dataset]
        train_data, test_data, val_data = load_fn(test, data_config)

        # use DatasetClass to preprocess the data
        train_data, test_data, val_data = DatasetClass(train_data, tokenizer), DatasetClass(test_data, tokenizer), DatasetClass(val_data, tokenizer)
    
    else: # GPT2 uses different preprocessing
        loaders = {
            'squad': (load_squad, preprocess_squad),
            'wmt16_en_de': (load_wmt, preprocess_wmt),
            'imdb': (load_imdb, preprocess_imdb),
        }
        load_fn, preprocess_fn = loaders[dataset]
        train_data, test_data, val_data = load_fn(test, data_config)

        # map using preprocessing function
        train_data = train_data.map(
            lambda x: preprocess_fn(x, tokenizer),
            batched=True, 
            remove_columns=train_data.column_names
        )
        test_data = test_data.map(
            lambda x: preprocess_fn(x, tokenizer),
            batched=True, 
            remove_columns=test_data.column_names
        )
        val_data = val_data.map(
            lambda x: preprocess_fn(x, tokenizer),
            batched=True, 
            remove_columns=val_data.column_names
        )

    if dataset not in loaders:
        raise NotImplementedError(
            f"Dataset '{dataset}' not implemented."
            f"Available options: {list(loaders.keys())}"
        )

    return train_data, test_data, val_data

# ===============================================================================================

class BaseTrainer:
    """
    Base trainer class for fine-tuning language models.
    
    This class handles the complete training pipeline including:
    - Model and dataset initialization
    - Weights & Biases integration
    - Training configuration
    - Model saving
    
    Attributes:
        device (str): Device to run training on
        model_name (str): Name of the model being trained
        dataset_name (str): Name of the dataset being used
        finetune_type (str): Type of fine-tuning being used
        train_batch_size (int): Batch size for training
        eval_batch_size (int): Batch size for evaluation
        model: The model being trained
        tokenizer: The tokenizer being used
        train_data: Training dataset
        test_data: Test dataset
        val_data: Validation dataset
        default_lr (dict): Default learning rates for different fine-tuning types
    """
    def __init__(self, device, model, dataset, finetune, train_batch_size, eval_batch_size, test=False):
        """
        Initialize the trainer with model and dataset configurations.
        
        Args:
            device (str): Device to run training on ('cpu' or 'cuda')
            model (str): Name of the model to train
            dataset (str): Name of the dataset to use
            finetune (str): Type of fine-tuning to use
            train_batch_size (int): Batch size for training
            eval_batch_size (int): Batch size for evaluation
            test (bool, optional): Whether to use test mode. Defaults to False.
        """
        self.device = device
        self.model_name = model
        self.dataset_name = dataset
        self.finetune_type = finetune
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.model, self.tokenizer = get_model_tokenizer(model, finetune, data2task[dataset], self.device)
        self.train_data, self.test_data, self.val_data = get_dataset(dataset, self.tokenizer, model, test)
        self.default_lr = {
            "full": 5e-5,
            "lora": 3e-4,
            "adapters": 1e-6
        }

        debug_print(
            title="Trainer intialization",
            model=self.model_name,
            dataset=self.dataset_name,
            train_size=self.train_data.__len__(),
            test_size=self.test_data.__len__(),
            val_size=self.val_data.__len__(),
            finetune_type=self.finetune_type,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            device=self.device,
        )

        # Ensure correct model and tokenizer are loaded
        # debug_print(title="Model architecture", model=self.model)
        # debug_print(title="Tokenizer architecture", tokenizer=self.tokenizer)

    def set_wandb_api(self, wandb_token, wandb_api, project):
        """
        Configure Weights & Biases API credentials.
        
        Args:
            wandb_token (str): HuggingFace token for W&B
            wandb_api (str): W&B API key
            project (str): W&B project name
        """
        self.wandb_token = wandb_token
        self.wandb_api = wandb_api
        self.project = project

    def login_wandb(self, project, name):
        """
        Log in to Weights & Biases and set up environment variables.
        
        Args:
            project (str): W&B project name
            name (str): Name for the current run
        """
        os.environ.update({
            "HUGGINGFACE_TOKEN": self.wandb_token,
            "WANDB_API_KEY": self.wandb_api,
            "WANDB_PROJECT": project,
            "WANDB_NAME": name
        })
        login(self.wandb_token)

    def get_training_args(self, num_train_epochs, learning_rate, weight_decay, logging_steps, use_cpu):
        """
        Configure training arguments for the model.
        
        Args:
            num_train_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for training
            weight_decay (float): Weight decay for regularization
            logging_steps (int): Number of steps between logging
            use_cpu (bool): Whether to force CPU usage
            
        Returns:
            TrainingArguments: Configured training arguments
        """
        return TrainingArguments(
            output_dir=os.getenv("WANDB_NAME"),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            warmup_steps=100,
            weight_decay=weight_decay,
            logging_dir=os.path.join(os.getenv("WANDB_NAME"), "logs"),
            # logging_steps=logging_steps,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=os.getenv("WANDB_NAME"),
            fp16=True,
            gradient_accumulation_steps=4,
            save_total_limit=2,
            remove_unused_columns=False,
            use_cpu=use_cpu
        )
    
    def get_trainer(self, args, data_collator):
        """
        Get the appropriate trainer for the model and task.
        
        Args:
            args (TrainingArguments): Training configuration
            data_collator: Data collator for batching
            
        Returns:
            Trainer: Configured trainer instance
        """
        # Use extractive QA trainer for GPT-2
        if self.model_name == 'gpt2' and self.dataset_name == 'squad':
            debug_print(title='Using ExtractiveQATrainer')
            return ExtractiveQATrainer(
                model=self.model, 
                args=args, 
                train_dataset=self.train_data, 
                eval_dataset=self.val_data, 
                data_collator=data_collator,
            )
        
        debug_print(title='Using Hugging Face Trainer')
        return Trainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset=self.val_data, 
            args=args, 
            data_collator=data_collator,
        )

    def train_and_save(self, trainer, saved_model):
        """
        Train the model and save the results.
        
        Args:
            trainer: The trainer instance to use
            saved_model (str): Path to save the model
        """
        # Start training
        trainer.train()
        trainer.save_model(saved_model)
        self.tokenizer.save_pretrained(saved_model)

        print(f'Finetuned model and tokenizer saved to {saved_model}.')

    def run(self, saved_model, num_train_epochs=10, learning_rate=None, weight_decay=0.02, logging_steps=1, use_cpu=False):
        """
        Run the complete training pipeline.
        
        Args:
            saved_model (str): Path to save the model
            num_train_epochs (int, optional): Number of training epochs. Defaults to 10.
            learning_rate (float, optional): Learning rate. Defaults to None.
            weight_decay (float, optional): Weight decay. Defaults to 0.02.
            logging_steps (int, optional): Logging frequency. Defaults to 1.
            use_cpu (bool, optional): Force CPU usage. Defaults to False.
        """
        debug_print(
            title="Number of training epochs",
            num_train_epochs=num_train_epochs
        )

        self.login_wandb(self.project, saved_model)

        # Set default learning rate per fine-tune type
        learning_rate = learning_rate if learning_rate else self.default_lr[self.finetune_type]

        # Setting up training arguments
        args = self.get_training_args(num_train_epochs, learning_rate, weight_decay, logging_steps, use_cpu)

        # Seq2Seq collator
        if 'gpt' not in self.model_name:
            debug_print(title='Using DataCollatorForSeq2Seq')
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True,  # or "longest"
                return_tensors="pt"
            )
        else:
            debug_print(title='Using DataCollatorForLanguageModeling')
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )


        # Start training loop
        trainer = self.get_trainer(args, data_collator)

        # Save finetuned model
        self.train_and_save(trainer, saved_model)
