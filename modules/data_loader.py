import torch
from datasets import load_dataset
from typing import Tuple, Callable, Dict
from preprocess_loader import preprocess_squad, preprocess_wmt, preprocess_imdb


# ============================= GENERIC DATA LOADER ============================= #

def load_dataset_wrapper(name: str, config: str = None, test: bool = False) -> Tuple:
    """
    Generic loader for HuggingFace datasets with optional test mode.
    """
    dataset = load_dataset(name, config) if config else load_dataset(name)

    if test:
        dataset = {
            'train': dataset['train'].select(range(20)),
            'validation': dataset['validation'].select(range(20))
        }

    train_test = dataset['train'].train_test_split(test_size=0.2, seed=42)
    return train_test['train'], train_test['test'], dataset['validation']


def load_squad(test: bool = False):
    return load_dataset_wrapper('squad', test=test)

def load_wmt(test: bool = False):
    return load_dataset_wrapper('wmt16', config='de-en', test=test)

def load_imdb(test: bool = False):
    return load_dataset_wrapper('imdb', test=test)


# ============================= BASE DATASET CLASS ============================= #

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        preprocess_fn: Callable,
        max_input_length: int = 512,
        max_target_length: int = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Preprocessing
        if max_target_length is not None:
            self.model_inputs = preprocess_fn(dataset, tokenizer, max_input_length, max_target_length)
        else:
            self.model_inputs = preprocess_fn(dataset, tokenizer, max_input_length)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.dataset)


# ============================= TASK-SPECIFIC WRAPPERS ============================= #

class SquadDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
        super().__init__(dataset, tokenizer, preprocess_squad, max_input_length=max_input_length, max_target_length=max_target_length)

class WMTDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
        super().__init__(dataset, tokenizer, preprocess_wmt, max_input_length, max_target_length)

class IMDBDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, max_input_length: int = 512, max_target_length: int = 16):
        super().__init__(dataset, tokenizer, preprocess_imdb, max_input_length, max_target_length)
