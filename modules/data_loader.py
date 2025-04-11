import torch
from datasets import load_dataset
from preprocess_loader import preprocess_squad, preprocess_wmt, preprocess_imdb

# ============================= DATASET LOADER ============================= #

def load_squad(test=False):
    """
    Args:
        test (Bool): set test=True to test the pipeline with only 20 samples.
    Returns:
        train_data (Dataset): the train dataset.
        test_data (Dataset): the test dataset.
        val_data (Dataset):  the validation dataset.
    """

    # Load SQuAD dataset
    dataset = load_dataset('squad')

    # Select only the first 20 examples using .select() to keep it as Dataset
    if test == True:
        dataset = {
            'train': dataset['train'].select(range(20)),
            'validation': dataset['validation'].select(range(20))
        }

    # Split train set into train and test (80-20 split)
    train_test = dataset['train'].train_test_split(test_size=0.2, seed=42)

    train_data = train_test['train']
    test_data = train_test['test']
    val_data = dataset['validation']

    return train_data, test_data, val_data


def load_wmt(test=False):
    """
    Args:
        test (Bool): set test=True to test the pipeline with only 20 samples.
    Returns:
        train_data (Dataset): the train dataset.
        test_data (Dataset): the test dataset.
        val_data (Dataset):  the validation dataset.
    """

    # Load WMT16 English-German dataset
    dataset = load_dataset("wmt16", "de-en")

    # Select only the first 20 examples using .select() to keep it as Dataset
    if test == True:
        dataset = {
            'train': dataset['train'].select(range(20)),
            'validation': dataset['validation'].select(range(20))
        }

    # Split train set into train and test (80-20 split)
    train_test = dataset['train'].train_test_split(test_size=0.2, seed=42)

    train_data = train_test['train']
    test_data = train_test['test']
    val_data = dataset['validation']

    return train_data, test_data, val_data

def load_imdb(test=False):
    """
    Args:
        test (Bool): set test=True to test the pipeline with only 20 samples.
    Returns:
        train_data (Dataset): the train dataset.
        test_data (Dataset): the test dataset.
        val_data (Dataset):  the validation dataset.
    """

    # Load WMT16 English-German dataset
    dataset = load_dataset("imdb")

    # Select only the first 20 examples using .select() to keep it as Dataset
    if test == True:
        dataset = {
            'train': dataset['train'].select(range(20)),
            'validation': dataset['validation'].select(range(20))
        }

    # Split train set into train and test (80-20 split)
    train_test = dataset['train'].train_test_split(test_size=0.2, seed=42)

    train_data = train_test['train']
    test_data = train_test['test']
    val_data = dataset['validation']

    return train_data, test_data, val_data

# ============================= DATASET MODULE ============================= #

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.model_inputs = preprocess_squad(self.dataset, self.tokenizer, self.max_length)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self.dataset)
    

class WMTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = preprocess_wmt(self.dataset, self.tokenizer, self.max_input_length, self.max_target_length)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self.dataset)
    
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=16):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = preprocess_imdb(self.dataset, self.tokenizer, self.max_input_length, self.max_target_length)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self.dataset)