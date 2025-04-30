# For training
from transformers import (
    TrainingArguments, Trainer, # For training
    DataCollatorForSeq2Seq
)
from modules.train.custom_trainer import ExtractiveQATrainer, LeastTrainLossTrainer# For GPT-2 QA trainer
from modules.data.data_collator import FastDataCollatorForSeq2Seq

# For login wandb
import os
from huggingface_hub import login

# For preprocessing
from modules.data.hfdata import load_squad, load_wmt, load_imdb, SquadDataset, SquadDatasetExtractive, WMTDataset, IMDBDataset
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
    loaders = {
        't5-base': load_t5_base,
        'bart-base': load_bart_base,
        'gpt2': load_gpt_2
    }

    # print(f'Loading {model} for {finetune_type}.')

    for key in loaders:
        if key in model:
            return loaders[key](model, finetune_type, task, device)

    raise NotImplementedError(
        f"Model '{model}' not implemented."
        f"Available options: {list(loaders.keys())}"
    )
    
def get_dataset(dataset, tokenizer, model_name, test):
    # get config from file
    with open('modules/data/config.json', 'r', encoding='utf-8') as file:
        data_config = json.load(file)
    
    # generic loader
    loaders = {
        'squad': (load_squad, SquadDataset) if 'gpt' not in model_name else (load_squad, SquadDatasetExtractive),
        'wmt16_en_de': (load_wmt, WMTDataset),
        'imdb': (load_imdb, IMDBDataset)
    }

    if dataset not in loaders:
        raise NotImplementedError(
            f"Dataset '{dataset}' not implemented."
            f"Available options: {list(loaders.keys())}"
        )

    load_fn, DatasetClass = loaders[dataset]
    train_data, test_data, val_data = load_fn(test, data_config)
    return DatasetClass(train_data, tokenizer), DatasetClass(test_data, tokenizer), DatasetClass(val_data, tokenizer)

# ===============================================================================================

class BaseTrainer:
    def __init__(self, device, model, dataset, finetune, train_batch_size, eval_batch_size, test=False):
        self.device = device
        self.model_name = model
        self.dataset_name = dataset
        self.finetune_type = finetune
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.model, self.tokenizer = get_model_tokenizer(model, finetune, data2task[dataset], self.device)
        self.train_data, self.test_data, self.val_data = get_dataset(dataset, self.tokenizer, model, test)
        
        print(f"""================================ Model architecture ==================================
{self.model}
====================================================================================
""")

    def set_wandb_api(self, wandb_token, wandb_api, project):
        self.wandb_token = wandb_token
        self.wandb_api = wandb_api
        self.project = project

    def login_wandb(self, project, name):
        os.environ.update({
            "HUGGINGFACE_TOKEN": self.wandb_token,
            "WANDB_API_KEY": self.wandb_api,
            "WANDB_PROJECT": project,
            "WANDB_NAME": name
        })
        login(self.wandb_token)

    def get_training_args(self, num_train_epochs, learning_rate, weight_decay, logging_steps, use_cpu):
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

    def train_and_save(self, trainer, saved_model):
        print(
    f"""================================ Start finetuning ==================================
****************** Finetune information ******************
- Model: {self.model_name}
- Dataset: {self.dataset_name}
- Finetune strategy: {self.finetune_type}
====================================================================================
""")
        trainer.train()
        trainer.save_model(saved_model)
        self.tokenizer.save_pretrained(saved_model)

        print(f'Finetuned model and tokenizer saved to {saved_model}.')

    def run(self, saved_model, num_train_epochs=10, learning_rate=None, weight_decay=0.02, logging_steps=1, use_cpu=False):
        print(f"""================================  Training information ==================================
- Using device: {self.device}
- No. epoch(s): {num_train_epochs}
- Train batch size: {self.train_batch_size}
- Eval batch size: {self.eval_batch_size}
=========================================================================================
""")

        self.login_wandb(self.project, saved_model)

        # Set default learning rate per fine-tune type
        if learning_rate is None:
            if self.finetune_type == "full":
                learning_rate = 5e-5
            elif self.finetune_type == "lora":
                learning_rate = 3e-4
            elif self.finetune_type == "adapters":
                learning_rate = 1e-6

        # Setting up training arguments
        args = self.get_training_args(num_train_epochs, learning_rate, weight_decay, logging_steps, use_cpu)

        # Seq2Seq collator
        # data_collator = DataCollatorForSeq2Seq(
        #     tokenizer=self.tokenizer,
        #     model=self.model,
        #     padding=True,  # or "longest"
        #     return_tensors="pt"
        # )

        data_collator = FastDataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

        # Start training loop
        if self.model_name == 'gpt2' and self.dataset_name == 'squad':
            # Use custom trainer for GPT-2 extractive QA
            trainer = ExtractiveQATrainer(
                model=self.model, 
                args=args, 
                train_dataset=self.train_data, 
                eval_dataset=self.val_data, 
                data_collator=data_collator,
                # compute_metrics=compute_metrics,
                # optimizers=(optimizer, None),  # Custom optimizer, no scheduler
            )
        elif self.model_name == 't5-base' and self.dataset_name == 'imdb':
            # Use custom trainer getting least train loss
            trainer = Trainer( # working around...
                model=self.model, 
                args=args, 
                train_dataset=self.train_data, 
                eval_dataset=self.val_data, 
                data_collator=data_collator,
                # compute_metrics=compute_metrics,
                # optimizers=(optimizer, None),  # Custom optimizer, no scheduler
            )
        else:
            # Use default trainer for other models and datasets
            trainer = Trainer(
                model=self.model, 
                args=args, 
                train_dataset=self.train_data, 
                eval_dataset=self.val_data, 
                data_collator=data_collator,
                # compute_metrics=compute_metrics,
                # optimizers=(optimizer, None),  # Custom optimizer, no scheduler
            )

        # Save finetuned model
        self.train_and_save(trainer, saved_model)
