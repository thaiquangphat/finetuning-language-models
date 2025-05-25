from modules.data.hfdatasets.squad import prepare_squad, prepare_squad_decoder, prepare_squad_extractive
from modules.data.hfdatasets.wmt import prepare_wmt, prepare_wmt_decoder
from modules.data.hfdatasets.imdb import prepare_imdb, prepare_imdb_decoder
from modules.train.ultis import debug_print # For debugging
import torch

class SquadDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling SQuAD question answering data.
    
    This class processes the SQuAD dataset for sequence-to-sequence question answering,
    handling tokenization and formatting of inputs and targets for model training.
    
    Args:
        dataset: The raw SQuAD dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 128)
    """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence
                - attention_mask (torch.Tensor): Attention mask for input sequence
                - labels (torch.Tensor): Tokenized target sequence
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class SquadDatasetExtractive(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for extractive question answering on SQuAD data.
    
    This class processes the SQuAD dataset for extractive question answering,
    handling tokenization and formatting of inputs and answer span positions.
    
    Args:
        dataset: The raw SQuAD dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 128)
    """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence
                - attention_mask (torch.Tensor): Attention mask for input sequence
                - start_positions (torch.Tensor): Start position of answer span
                - end_positions (torch.Tensor): End position of answer span
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "start_positions": torch.tensor(self.model_inputs["start_positions"][idx], dtype=torch.long),
            "end_positions": torch.tensor(self.model_inputs["end_positions"][idx], dtype=torch.long),
        }

class SquadDatasetDecoder(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for decoder-based question answering on SQuAD data.
    
    This class processes the SQuAD dataset for decoder-based question answering,
    handling tokenization and formatting of inputs and targets for decoder models.
    
    Args:
        dataset: The raw SQuAD dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence
                - attention_mask (torch.Tensor): Attention mask for input sequence
                - labels (torch.Tensor): Tokenized target sequence
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class WMTDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling WMT16 English-German translation data.
    
    This class processes the WMT16 dataset for machine translation,
    handling tokenization and formatting of inputs and targets for model training.
    
    Args:
        dataset: The raw WMT16 dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence
                - attention_mask (torch.Tensor): Attention mask for input sequence
                - labels (torch.Tensor): Tokenized target sequence
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class WMTDatasetDecoder(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for decoder-based translation on WMT16 data.
    
    This class processes the WMT16 dataset for decoder-based translation,
    handling tokenization and formatting of inputs and targets for decoder models.
    
    Args:
        dataset: The raw WMT16 dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence
                - attention_mask (torch.Tensor): Attention mask for input sequence
                - labels (torch.Tensor): Tokenized target sequence
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class IMDBDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling IMDb sentiment analysis data.
    
    This class processes the IMDb dataset for sentiment analysis,
    handling tokenization and formatting of inputs and targets for model training.
    
    Args:
        dataset: The raw IMDb dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence
                - attention_mask (torch.Tensor): Attention mask for input sequence
                - labels (torch.Tensor): Tokenized target sequence
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }
    
class IMDBDatasetDecoder(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for decoder-based sentiment analysis on IMDb data.
    
    This class processes the IMDb dataset for decoder-based sentiment analysis,
    handling tokenization and formatting of inputs and targets for decoder models.
    
    Args:
        dataset: The raw IMDb dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    """
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
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence
                - attention_mask (torch.Tensor): Attention mask for input sequence
                - labels (torch.Tensor): Tokenized target sequence
        """
        return {
            "input_ids": torch.tensor(self.model_inputs["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.model_inputs["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.model_inputs["labels"][idx], dtype=torch.long),
        }