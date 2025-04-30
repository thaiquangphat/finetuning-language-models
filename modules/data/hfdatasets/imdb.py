from datasets import load_dataset, concatenate_datasets
from typing import Dict, Tuple, Callable
import re

def load_imdb(test: bool = False, data_config: Dict = None) -> Tuple:
    """
    Loads the IMDb dataset and splits the train set into train and validation.

    Args:
        test (bool): If True, uses only 20 samples from each split for testing pipeline.
        data_config (Dict): contains portion of data samples to get from the original dataset.
    Returns:
        train_data (Dataset): the train dataset.
        test_data (Dataset): the test dataset.
        val_data (Dataset):  the validation dataset.
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
        max_target_length=64
    ):
    """
    Preprocessing function for IMDb sentiment analysis dataset, compatible with batched processing.
    
    Args:
        examples (Dict): Batch of dataset examples with 'text' and 'label' fields.
        max_input_length (int): Maximum number of tokens for input (review) sequences.
        max_target_length (int): Maximum number of tokens for target (sentiment) sequences.
    
    Returns:
        model_inputs (Dict): Input for model training with tokenized inputs and labels.
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


