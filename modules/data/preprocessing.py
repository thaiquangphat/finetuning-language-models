# ============================= PREPROCESSING FUNCTION ============================= #
def generate_inputs(tokenizer, inputs, targets, max_input_length=512, max_target_length=128):
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding='max_length'
    )

    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding='max_length'
    )
    
    # Replace padding token id in labels with -100 to ignore in loss
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_squad(dataset, tokenizer, max_input_length=512, max_target_length=128):
    """
    Preprocessing function:
    Args:
        dataset (Dataset): input dataset.
        tokenizer (AutoTokenizer): tokenizer.
        max_length (int: 1024): maximum number of tokens allowed in the input sequence. 

    Returns:
        model_inputs (Dict): input for model training.
    """

    inputs = ["question: " + q + " context: " + c for q, c in zip(dataset["question"], dataset["context"])]
    targets = [a["text"][0] for a in dataset["answers"]]  # Take first answer only
    
    return generate_inputs(tokenizer, inputs, targets, max_input_length, max_target_length)

def preprocess_wmt(dataset, tokenizer, max_input_length=512, max_target_length=128):
    """
    Preprocessing function for WMT English-German translation dataset.
    
    Args:
        dataset (Dataset): Input dataset (e.g., wmt16 train/validation split).
        tokenizer (ProphetNetTokenizer): Tokenizer for ProphetNet.
        max_input_length (int): Maximum number of tokens for input (English) sequences.
        max_target_length (int): Maximum number of tokens for target (German) sequences.
    
    Returns:
        model_inputs (Dict): Input for model training with tokenized inputs and labels.
    """
    # Extract English and German texts
    inputs = [f'english: {data["translation"]["en"]}' for data in dataset]
    targets = [f'german: {data["translation"]["de"]}' for data in dataset]
    
    return generate_inputs(tokenizer, inputs, targets, max_input_length, max_target_length)

def preprocess_imdb(dataset, tokenizer, max_input_length=512, max_target_length=16):
    """
    Preprocessing function for IMDb sentiment analysis dataset, compatible with batched processing.
    
    Args:
        examples (Dict): Batch of dataset examples with 'text' and 'label' fields.
        max_input_length (int): Maximum number of tokens for input (review) sequences.
        max_target_length (int): Maximum number of tokens for target (label) sequences.
    
    Returns:
        model_inputs (Dict): Tokenized inputs with input_ids, attention_mask, and labels.
    """
    # Extract review texts and labels
    inputs = dataset["text"]
    # Convert numeric labels (0, 1) to text ("negative", "positive")
    targets = ["positive" if label == 1 else "negative" for label in dataset["label"]]
    
    return generate_inputs(tokenizer, inputs, targets, max_input_length, max_target_length)