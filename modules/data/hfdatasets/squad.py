from datasets import load_dataset
from typing import Dict

# ============================ LOADING DATA ============================= #

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

def prepare_squad(
        dataset, 
        tokenizer, 
        max_input_length=512, 
        max_target_length=128
    ):
    
    # Formating inputs in string format
    inputs = ["question: " + q + " context: " + c for q, c in zip(dataset['question'], dataset['context'])]
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

def prepare_squad_extractive(
        dataset, 
        tokenizer, 
        max_input_length=512, 
        max_target_length=128
    ):
    """
    Preprocessing function for extractive tasks.
    
    Args:
        dataset (Dataset): Input dataset (e.g., SQuAD).
        tokenizer (AutoTokenizer): Tokenizer for the model.
        max_input_length (int): Maximum number of tokens for input sequences.
        max_target_length (int): Maximum number of tokens for target sequences.
    
    Returns:
        model_inputs (Dict): Input for model training with tokenized inputs and labels.
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