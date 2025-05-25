from datasets import load_dataset, concatenate_datasets
from typing import Dict, Tuple, Callable
import re

def load_imdb(test: bool = False, data_config: Dict = None) -> Tuple:
    """
    Loads the IMDb dataset and splits it into train, validation, and test sets.
    
    This function loads the IMDb dataset from HuggingFace datasets, applies the specified
    data portion configuration, and creates balanced train/validation/test splits.
    The validation set is created by taking a portion of the test set to ensure
    balanced class distribution.
    
    Args:
        test (bool): If True, uses only 20 samples from each split for testing pipeline
        data_config (Dict): Configuration dictionary containing data portion settings
            - imdb.train_portion: Portion of training data to use
            - imdb.val_portion: Portion of validation data to use
    
    Returns:
        tuple: A tuple containing:
            - train_data (Dataset): Training dataset
            - test_data (Dataset): Test dataset
            - val_data (Dataset): Validation dataset
    """

    imdb_dataset = load_dataset('imdb')

    # Shuffle test set to avoid label order bias
    test_shuffled = imdb_dataset['test'].shuffle(seed=42)

    # Separate by label
    test_neg = test_shuffled.filter(lambda x: x['label'] == 0)
    test_pos = test_shuffled.filter(lambda x: x['label'] == 1)

    # Get 10% of each for validation
    n_val_neg = int(0.1 * len(test_neg))
    n_val_pos = int(0.1 * len(test_pos))

    val_neg = test_neg.select(range(n_val_neg))
    val_pos = test_pos.select(range(n_val_pos))
    validation_set = concatenate_datasets([val_neg, val_pos]).shuffle(seed=42)

    # Use remaining test samples for training
    train_extra = concatenate_datasets([
        test_neg.select(range(n_val_neg, len(test_neg))),
        test_pos.select(range(n_val_pos, len(test_pos)))
    ])

    # Final splits
    imdb_dataset = {
        'train': concatenate_datasets([imdb_dataset['train'], train_extra]).shuffle(seed=42),
        'validation': validation_set
    }

    # Select range based on config
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

def prepare_imdb(
        dataset, 
        tokenizer, 
        max_input_length=1024, 
        max_target_length=1024
    ):
    """
    Prepares IMDb dataset for sequence-to-sequence sentiment analysis.
    
    This function formats the input data by combining movie reviews with sentiment prompts,
    cleans the text, and prepares the sentiment labels for training.
    
    Args:
        dataset: The IMDb dataset containing movie reviews and sentiment labels
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 1024)
        max_target_length (int): Maximum length of target sequences (default: 1024)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for input sequences
            - labels: Tokenized target sequences with padding tokens replaced by -100
    """

    def clean_imdb(text):
        # Lowercase the text
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Extract review texts and labels
    inputs = ["determine text sentiment: " + clean_imdb(text) for text in dataset["text"]]
    # Convert numeric labels (0, 1) to text ("negative", "positive")
    targets = ["sentiment: " + ("True" if label == 1 else "False") for label in dataset["label"]]
    
    # Generating model inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding='max_length',
        return_tensors="pt"
    )

    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding='max_length',
        return_tensors="pt"
    )

    # Replace padding token id in labels with -100 to ignore in loss
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def prepare_imdb_decoder(
        dataset, 
        tokenizer, 
        max_input_length=1024, 
        max_target_length=1024
    ):
    """
    Prepares IMDb dataset for decoder-based sentiment analysis.
    
    This function formats the input data for decoder models by combining
    movie reviews and sentiment labels into a single sequence.
    
    Args:
        dataset: The IMDb dataset containing movie reviews and sentiment labels
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 1024)
        max_target_length (int): Maximum length of target sequences (default: 1024)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for input sequences
            - labels: Tokenized target sequences
    """

    def clean_imdb(text):
        # Lowercase the text
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Extract review texts and labels
    inputs = ["determine text sentiment: " + clean_imdb(text) for text in dataset["text"]]
    labels = ["sentiment: " + ("True" if label == 1 else "False") for label in dataset["label"]]
    targets = [f'{inp} {ans}' for inp, ans in zip(inputs, labels)]
    
    # Generating model inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding='max_length',
        return_tensors="pt"
    )

    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding='max_length',
        return_tensors="pt"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_imdb(
        example,
        tokenizer,
        max_input_length=512,
        max_target_length=128,
    ):
    """
    Preprocesses a single IMDb example for model training.
    
    This function processes a single example from the IMDb dataset,
    cleaning the text and formatting it for sequence-to-sequence training
    with proper masking.
    
    Args:
        example: A single example from the IMDb dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 128)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized and padded input sequence
            - attention_mask: Attention mask for the sequence
            - labels: Tokenized target sequence with proper masking
    """
    def clean_imdb(text):
        # Lowercase the text
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Extract fields
    text = example['text']
    sentiment = "True" if example['label'] == 1 else "False"

    # Create prompt and target
    prompt = "determine text sentiment: " + clean_imdb(text) + "sentiment: "
    target = f" {sentiment}"

    # Tokenize prompt and target
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_input_length - max_target_length, padding=False).input_ids
    target_ids = tokenizer(target, truncation=True, max_length=max_target_length, padding=False).input_ids

    # Combine inputs and create labels with masking
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    # Truncate/pad to max_input_length
    input_ids = input_ids[:max_input_length]
    labels = labels[:max_input_length]

    attention_mask = [1] * len(input_ids)

    # Pad if needed
    pad_len = max_input_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }