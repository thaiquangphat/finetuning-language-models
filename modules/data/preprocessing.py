import re

# ============================= GENERATING INPUTS FUNCTION ============================= #
def generate_inputs(tokenizer, inputs, targets, max_input_length=512, max_target_length=128):
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

# ============================= PREPROCESSING FUNCTION ============================= #

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

def preprocess_squad_gpt2(dataset, tokenizer, max_input_length=512, max_target_length=128):
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

    inputs = ["question: " + q + " context: " + c for q, c in zip(dataset["question"], dataset["context"])]

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
    inputs = [f'translate to german. english: {data["translation"]["en"]}' for data in dataset]
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
    inputs = ["sentiment analysis: " + clean_imdb(text) for text in dataset["text"]]
    # Convert numeric labels (0, 1) to text ("negative", "positive")
    targets = ["positive" if label == 1 else "negative" for label in dataset["label"]]
    print(targets[0])
    
    return generate_inputs(tokenizer, inputs, targets, max_input_length, max_target_length)
