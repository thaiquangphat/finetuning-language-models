# For training
from transformers import (
    TrainingArguments, Trainer # For training
)
from peft import LoraConfig, get_peft_model, TaskType
from adapters import AdapterConfig, init

# For login wandb
import os
from huggingface_hub import login

# For preprocessing
from modules.datasets import load_squad, load_wmt, load_imdb, SquadDataset, WMTDataset, IMDBDataset
from modules.models import load_t5_base, load_bart_base, load_prophetnet_large

# ===============================================================================================
def get_model_tokenizer(model, device):
    loaders = {
        't5-base': load_t5_base,
        'bart-base': load_bart_base,
        'prophetnet-large-uncased': load_prophetnet_large
    }

    for key in loaders:
        if key in model:
            return loaders[key](model, device)

    raise NotImplementedError(
        f"Model '{model}' not implemented."
        f"Available options: {list(loaders.keys())}"
    )
    
def get_dataset(dataset, tokenizer, test):
    loaders = {
        'squad': (load_squad, SquadDataset),
        'wmt16_en_de': (load_wmt, WMTDataset),
        'imdb': (load_imdb, IMDBDataset)
    }

    if dataset not in loaders:
        raise NotImplementedError(
            f"Dataset '{dataset}' not implemented."
            f"Available options: {list(loaders.keys())}"
        )

    load_fn, DatasetClass = loaders[dataset]
    train_data, test_data, val_data = load_fn(test)
    return DatasetClass(train_data, tokenizer), DatasetClass(test_data, tokenizer), DatasetClass(val_data, tokenizer)

# ===============================================================================================

class BaseTrainer:
    def __init__(self, device, model, dataset, finetune, test=False):
        self.device = device
        self.model_name = model
        self.dataset_name = dataset
        self.finetune_type = finetune
        self.model, self.tokenizer = get_model_tokenizer(model, self.device)
        self.train_data, self.test_data, self.val_data = get_dataset(dataset, self.tokenizer, test)

    def login_wandb(self, project, name, token="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", api_key="d83175b72ab7d073e2ed4f0e60ef001c11cd4555"):
        os.environ.update({
            "HUGGINGFACE_TOKEN": token,
            "WANDB_API_KEY": api_key,
            "WANDB_PROJECT": project,
            "WANDB_NAME": name
        })
        login(token)

    def get_training_args(self, num_train_epochs, learning_rate, weight_decay, logging_steps):
        return TrainingArguments(
            output_dir=os.getenv("WANDB_NAME"),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=weight_decay,
            logging_dir=os.path.join(os.getenv("WANDB_NAME"), "logs"),
            logging_steps=logging_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=os.getenv("WANDB_NAME")
        )

    def apply_finetune_strategy(self):
        if 't5-base' in self.model_name:
            target_mods = ['q', 'v']
        elif 'bart-base' in self.model_name:
            target_mods = ["q_proj", "v_proj"]
        elif 'prohpetnet-large-uncased' in self.model_name:
            target_mods = ["query_proj", "value_proj"]
            
        if self.finetune_type == "lora":
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=target_mods,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)

        elif self.finetune_type == "adapters":
            init(self.model)
            adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=16, non_linearity="relu")
            self.model.add_adapter(self.dataset_name, config=adapter_config)
            self.model.train_adapter(self.dataset_name)
            self.model.set_active_adapters(self.dataset_name)

        # "full" fine-tuning does not require modification

    def train_and_save(self, trainer, saved_model):
        trainer.train()
        trainer.save_model(saved_model)
        self.tokenizer.save_pretrained(saved_model)

        print(f'Finetuned model and tokenizer saved to {saved_model}.')

    def run(self, saved_model, num_train_epochs=10, learning_rate=None, weight_decay=0.02, logging_steps=1):
        self.login_wandb('phat-ft-nlp', saved_model)

        # Set default learning rate per fine-tune type
        if learning_rate is None:
            if self.finetune_type == "full":
                learning_rate = 5e-5
            elif self.finetune_type == "lora":
                learning_rate = 3e-4
            elif self.finetune_type == "adapters":
                learning_rate = 1e-6

        # Applying finetune strategy
        self.apply_finetune_strategy()

        # Setting up training arguments
        args = self.get_training_args(num_train_epochs, learning_rate, weight_decay, logging_steps)

        # Start training loop
        trainer = Trainer(model=self.model, args=args, train_dataset=self.train_data, eval_dataset=self.val_data)

        # Save finetuned model
        self.train_and_save(trainer, saved_model)
