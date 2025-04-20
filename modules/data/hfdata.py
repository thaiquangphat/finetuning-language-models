import torch
from datasets import load_dataset, concatenate_datasets
from typing import Tuple, Callable, Dict
from modules.data.preprocessing import preprocess_squad, preprocess_wmt, preprocess_imdb


# ============================= GENERIC DATA LOADER ============================= #

def load_squad(test: bool = False, data_config: Dict = None):
    """
    Loads the Squad dataset and splits the train set into train and validation.

    Args:
        test (bool): If True, uses only 20 samples from each split for testing pipeline.
        data_config (Dict): contains portion of data samples to get from the original dataset.
    Returns:
        train_data (Dataset): the train dataset.
        test_data (Dataset): the test dataset.
        val_data (Dataset):  the validation dataset.
    """
    squad_dataset = load_dataset('squad')

    squad_dataset['train'] = squad_dataset['train'].select(range(int(data_config['squad']['train_portion'] * len(squad_dataset['train']))))
    squad_dataset['validation'] = squad_dataset['validation'].select(range(int(data_config['squad']['val_portion'] * len(squad_dataset['validation']))))

    split_squad = squad_dataset['train'].train_test_split(test_size=0.2, seed=42)

    squad_train = split_squad['train']
    squad_test  = split_squad['test']
    squad_val   = squad_dataset['validation']

    if test:
        squad_train = squad_train.select(range(20))
        squad_test = squad_test.select(range(20))
        squad_val = squad_val.select(range(20))

    return squad_train, squad_test, squad_val


def load_wmt(test: bool = False, data_config: Dict = None):
    """
    Loads the WMT16-En-De dataset and splits the train set into train and validation.

    Args:
        test (bool): If True, uses only 20 samples from each split for testing pipeline.
        data_config (Dict): contains portion of data samples to get from the original dataset.
    Returns:
        train_data (Dataset): the train dataset.
        test_data (Dataset): the test dataset.
        val_data (Dataset):  the validation dataset.
    """
    wmt_dataset = load_dataset('wmt16', 'de-en')

    wmt_dataset['train'] = wmt_dataset['train'].select(range(int(data_config['wmt16_en_de']['train_portion'] * len(wmt_dataset['train']))))
    wmt_dataset['validation'] = wmt_dataset['validation'].select(range(int(data_config['wmt16_en_de']['val_portion'] * len(wmt_dataset['validation']))))

    split_wmt = wmt_dataset['train'].train_test_split(test_size=0.2, seed=42)

    wmt_train = split_wmt['train']
    wmt_test  = split_wmt['test']
    wmt_val   = wmt_dataset['validation']

    if test:
        wmt_train = wmt_train.select(range(20))
        wmt_test = wmt_test.select(range(20))
        wmt_val = wmt_val.select(range(20))

    return wmt_train, wmt_test, wmt_val

def load_imdb(test: bool = False, data_config: Dict = None):
    """
    Loads the IMDB dataset and splits the train set into train and validation.

    Args:
        test (bool): If True, uses only 20 samples from each split for testing pipeline.
        data_config (Dict): contains portion of data samples to get from the original dataset.
    Returns:
        train_data (Dataset): the train dataset.
        test_data (Dataset): the test dataset.
        val_data (Dataset):  the validation dataset.
    """
    imdb_dataset = load_dataset('imdb')

    index = int(0.1 * len(imdb_dataset['test']))
    imdb_dataset = {
        'train': concatenate_datasets([imdb_dataset['train'], imdb_dataset['test'].select(range(index, len(imdb_dataset['test'])))]),
        'validation': imdb_dataset['test'].select(range(index))
    }

    imdb_dataset['train'] = imdb_dataset['train'].select(range(int(data_config['imdb']['train_portion'] * len(imdb_dataset['train']))))
    imdb_dataset['validation'] = imdb_dataset['validation'].select(range(int(data_config['imdb']['val_portion'] * len(imdb_dataset['validation']))))

    split_imdb = imdb_dataset['train'].train_test_split(test_size=0.2, seed=42)

    imdb_train = split_imdb['train']
    imdb_test  = split_imdb['test']
    imdb_val   = imdb_dataset['validation']

    if test:
        imdb_train = imdb_train.select(range(20))
        imdb_test = imdb_test.select(range(20))
        imdb_val = imdb_val.select(range(20))

    return imdb_train, imdb_test, imdb_val

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
        self.model_inputs = preprocess_fn(dataset, tokenizer, max_input_length, max_target_length)

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
        super().__init__(dataset, tokenizer, preprocess_squad, max_input_length, max_target_length)

class WMTDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
        super().__init__(dataset, tokenizer, preprocess_wmt, max_input_length, max_target_length)

class IMDBDataset(BaseDataset):
    def __init__(self, dataset, tokenizer, max_input_length: int = 512, max_target_length: int = 16):
        super().__init__(dataset, tokenizer, preprocess_imdb, max_input_length, max_target_length)
