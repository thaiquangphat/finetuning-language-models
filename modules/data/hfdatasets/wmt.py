from datasets import load_dataset, concatenate_datasets
from typing import Dict

def load_wmt(test: bool = False, data_config: Dict = None):
    """
    Loads the WMT16 English-German translation dataset and splits it into train, validation, and test sets.
    
    This function loads the WMT16 dataset from HuggingFace datasets, applies the specified
    data portion configuration, and splits the training data into train and test sets.
    
    Args:
        test (bool): If True, uses only 20 samples from each split for testing pipeline
        data_config (Dict): Configuration dictionary containing data portion settings
            - wmt16_en_de.train_portion: Portion of training data to use
            - wmt16_en_de.val_portion: Portion of validation data to use
    
    Returns:
        tuple: A tuple containing:
            - train_data (Dataset): Training dataset
            - test_data (Dataset): Test dataset
            - val_data (Dataset): Validation dataset
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

def prepare_wmt(
        dataset, 
        tokenizer, 
        max_input_length=512, 
        max_target_length=512
    ):
    """
    Prepares WMT16 dataset for sequence-to-sequence translation.
    
    This function formats the input data by combining English sentences with translation prompts,
    tokenizes them, and prepares the German translations as labels.
    
    Args:
        dataset: The WMT16 dataset containing English-German translation pairs
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for input sequences
            - labels: Tokenized target sequences with padding tokens replaced by -100
    """

    # Extract English and German texts
    inputs = [f'translate to german. english: {data["translation"]["en"]}' for data in dataset]
    targets = [f'german: {data["translation"]["de"]}' for data in dataset]
    
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

def prepare_wmt_decoder(
        dataset,
        tokenizer,
        max_input_length=512,
        max_target_length=512
    ):
    """
    Prepares WMT16 dataset for decoder-based translation.
    
    This function formats the input data for decoder models by combining
    English sentences and German translations into a single sequence.
    
    Args:
        dataset: The WMT16 dataset containing English-German translation pairs
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for input sequences
            - labels: Tokenized target sequences
    """
    # Extract English and German texts
    inputs = [f'translate to german. english: {data["translation"]["en"]}' for data in dataset]
    labels = [f'german: {data["translation"]["de"]}' for data in dataset]
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

def preprocess_wmt(
        example,
        tokenizer,
        max_input_length=512,
        max_target_length=128,
    ):
    """
    Preprocesses a single WMT16 example for model training.
    
    This function processes a single example from the WMT16 dataset,
    formatting it for sequence-to-sequence training with proper masking.
    
    Args:
        example: A single example from the WMT16 dataset
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 128)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized and padded input sequence
            - attention_mask: Attention mask for the sequence
            - labels: Tokenized target sequence with proper masking
    """
    # Extract fields
    english = example['translation']['en']
    german = example['translation']['de']

    # Create prompt and target
    prompt = f"translate to german. english: {english} german: "
    target = f" {german}"

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