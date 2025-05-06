from modules.data.hfdatasets.squad import prepare_squad, prepare_squad_decoder, prepare_squad_extractive
from modules.data.hfdatasets.wmt import prepare_wmt, prepare_wmt_decoder
from modules.data.hfdatasets.imdb import prepare_imdb, prepare_imdb_decoder
from modules.train.ultis import debug_print # For debugging
import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = prepare_squad(
            dataset, tokenizer, max_input_length, max_target_length
        )

        debug_print(title='Squad Dataset', task_type='question_answering', num_samples=self.dataset.__len__(), max_input_length=max_input_length, max_target_length=max_target_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns input_ids, attention_mask, and labels as tensors.
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class SquadDatasetExtractive(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = prepare_squad_extractive(
            dataset, tokenizer, max_input_length, max_target_length
        )

        debug_print(title='Squad Dataset Extractive', task_type='question_answering', num_samples=self.dataset.__len__(), max_input_length=max_input_length, max_target_length=max_target_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns input_ids, attention_mask, and start/end positions as tensors.
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "start_positions": torch.tensor(self.model_inputs["start_positions"][idx], dtype=torch.long),
            "end_positions": torch.tensor(self.model_inputs["end_positions"][idx], dtype=torch.long),
        }

class SquadDatasetDecoder(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = prepare_squad_decoder(
            dataset, tokenizer, max_input_length, max_target_length
        )

        debug_print(title='Squad Dataset Decoder', task_type='question_answering', num_samples=self.dataset.__len__(), max_input_length=max_input_length, max_target_length=max_target_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns input_ids, attention_mask, and labels as tensors.
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class WMTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = prepare_wmt(
            dataset, tokenizer, max_input_length, max_target_length
        )

        debug_print(title='WMT Dataset', task_type='translation', num_samples=self.dataset.__len__(), max_input_length=max_input_length, max_target_length=max_target_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns input_ids, attention_mask, and labels as tensors.
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class WMTDatasetDecoder(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = prepare_wmt_decoder(
            dataset, tokenizer, max_input_length, max_target_length
        )

        debug_print(title='WMT Dataset Decoder', task_type='translation', num_samples=self.dataset.__len__(), max_input_length=max_input_length, max_target_length=max_target_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns input_ids, attention_mask, and labels as tensors.
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = prepare_imdb(
            dataset, tokenizer, max_input_length, max_target_length
        )

        debug_print(title='IMDB Dataset', task_type='sentiment_analysis', num_samples=self.dataset.__len__(), max_input_length=max_input_length, max_target_length=max_target_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns input_ids, attention_mask, and labels as tensors.
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class IMDBDatasetDecoder(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.model_inputs = prepare_imdb_decoder(
            dataset, tokenizer, max_input_length, max_target_length
        )

        debug_print(title='IMDB Dataset Decoder', task_type='sentiment_analysis', num_samples=self.dataset.__len__(), max_input_length=max_input_length, max_target_length=max_target_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns input_ids, attention_mask, and labels as tensors.
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }