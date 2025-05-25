from datasets import load_dataset
from typing import Dict

# ============================ LOADING DATA ============================= #

def load_squad(test: bool = False, data_config: Dict = None):
    """
    Loads the SQuAD dataset and splits it into train, validation, and test sets.
    
    This function loads the SQuAD dataset from HuggingFace datasets, applies the specified
    data portion configuration, and splits the training data into train and test sets.
    
    Args:
        test (bool): If True, uses only 20 samples from each split for testing pipeline
        data_config (Dict): Configuration dictionary containing data portion settings
            - squad.train_portion: Portion of training data to use
            - squad.val_portion: Portion of validation data to use
    
    Returns:
        tuple: A tuple containing:
            - train_data (Dataset): Training dataset
            - test_data (Dataset): Test dataset
            - val_data (Dataset): Validation dataset
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

def prepare_squad(
        dataset, 
        tokenizer, 
        max_input_length=512, 
        max_target_length=128
    ):
    """
    Prepares SQuAD dataset for sequence-to-sequence question answering.
    
    This function formats the input data by combining questions and contexts,
    tokenizes them, and prepares the labels for training.
    
    Args:
        dataset: The SQuAD dataset containing questions, contexts, and answers
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 128)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for input sequences
            - labels: Tokenized target sequences with padding tokens replaced by -100
    """
    # Formating inputs in string format
    inputs = ["answer question: " + q + " context: " + c for q, c in zip(dataset['question'], dataset['context'])]
    targets = [a['text'][0] for a in dataset['answers']]  # Take first answer only

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

def prepare_squad_decoder(
        dataset,
        tokenizer,
        max_input_length=512,
        max_target_length=512
    ):
    """
    Prepares SQuAD dataset for decoder-based question answering.
    
    This function formats the input data for decoder models by combining
    questions, contexts, and answers into a single sequence.
    
    Args:
        dataset: The SQuAD dataset containing questions, contexts, and answers
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 512)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for input sequences
            - labels: Tokenized target sequences
    """
    # Formating inputs in string format
    inputs = ["answer question: " + q + " context: " + c for q, c in zip(dataset['question'], dataset['context'])]
    labels = [a['text'][0] for a in dataset['answers']]  # Take first answer only
    targets = [f'{inp} answer: {ans}' for inp, ans in zip(inputs, labels)]

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

def prepare_squad_extractive(
        dataset, 
        tokenizer, 
        max_input_length=512, 
        max_target_length=128
    ):
    """
    Prepares SQuAD dataset for extractive question answering.
    
    This function processes the dataset for extractive QA by identifying
    the start and end positions of answers within the context.
    
    Args:
        dataset: The SQuAD dataset containing questions, contexts, and answers
        tokenizer: The tokenizer to use for processing text
        max_input_length (int): Maximum length of input sequences (default: 512)
        max_target_length (int): Maximum length of target sequences (default: 128)
    
    Returns:
        dict: A dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for input sequences
            - start_positions: Start positions of answer spans
            - end_positions: End positions of answer spans
    """
    
    # Add PAD token to tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(
        dataset["question"],
        dataset["context"],
        max_length=max_input_length, 
        truncation=True, 
        padding='max_length',
        return_offsets_mapping=True,
    )

    offset_mapping = model_inputs.pop("offset_mapping")
    answers = dataset["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])

        sequence_ids = model_inputs.sequence_ids(i)
        
        # Identify where the context starts and ends within the tokenized sequence
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1
        
        start_pos = 0
        end_pos = 0
        
        # Check if the answer is within the context range
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            # Answer does not fit in the context
            start_positions.append(-100)
            end_positions.append(-100)
        else:
            for idx, (start, end) in enumerate(offsets):
                # Find the start position
                if start <= start_char < end:
                    start_pos = idx
                # Find the end position (break early once the end is found)
                if start < end_char <= end:
                    end_pos = idx
                    break
            
            # If end_pos is not found, we can default to start_pos (or handle as an edge case)
            if end_pos == 0:
                end_pos = start_pos
            
            start_positions.append(start_pos)
            end_positions.append(end_pos)

    # Add start and end positions to the model inputs
    model_inputs['start_positions'] = start_positions
    model_inputs['end_positions'] = end_positions

    return model_inputs

def preprocess_squad(
    example,
    tokenizer,
    max_input_length=512,
    max_target_length=128,
):
    """
    Preprocesses a single SQuAD example for model training.
    
    This function processes a single example from the SQuAD dataset,
    formatting it for sequence-to-sequence training with proper masking.
    
    Args:
        example: A single example from the SQuAD dataset
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
    context = example["context"]
    question = example["question"]
    answers = example["answers"]["text"]
    answer = answers[0] if answers else "No answer"

    # Create prompt and target
    prompt = f"answer question: {question} context: {context} answer:"
    target = f" {answer}"

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