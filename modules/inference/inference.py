import json
import torch
from tqdm import tqdm

def generate_output(model, tokenizer, input, device, max_length=1024):
    """
    Generates text output using a sequence-to-sequence model.
    
    This function takes an input text and generates a response using the provided model.
    It uses beam search with nucleus sampling for diverse and high-quality outputs.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for processing input/output text
        input (str): The input text to generate from
        device: The device to run inference on (e.g., 'cuda' or 'cpu')
        max_length (int, optional): Maximum length of input sequence. Defaults to 1024.
        
    Returns:
        str: The generated text response
    """
    input_text = input
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs, 
            max_length=256, # hardcode fix for bart-base adapters 
            num_beams=4, 
            early_stopping=True,
            repetition_penalty=2.0,  # Penalize repetition
            top_p=0.9,              # Nucleus sampling for diversity
            temperature=0.7,        # Control randomness
            no_repeat_ngram_size=3  # Prevent repeating n-grams
        ) #if not peft ==> generate(input, max_length=64, ...)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def generate_output_extractive(model, tokenizer, input, device, max_length=1024):
    """
    Generates extractive answer spans for question answering.
    
    This function takes a question and context, then predicts the start and end
    positions of the answer span within the context.
    
    Args:
        model: The extractive QA model
        tokenizer: The tokenizer for processing input text
        input (dict): Dictionary containing:
            - question (str): The question to answer
            - context (str): The context to extract answer from
        device: The device to run inference on (e.g., 'cuda' or 'cpu')
        max_length (int, optional): Maximum length of input sequence. Defaults to 1024.
        
    Returns:
        str: The extracted answer span
    """
    inputs = tokenizer(
        input['question'],
        input['context'],
        max_length=384,
        truncation="only_second",
        padding=max_length,
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_pred = torch.argmax(start_logits, dim=1).item()
        end_pred = torch.argmax(end_logits, dim=1).item()

    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokenizer.convert_tokens_to_string(tokens[start_pred:end_pred + 1])
    answer = answer.replace("[CLS]", "").replace("[SEP]", "").replace(" ##", "").strip()
    
    return answer

def run_inference(model, tokenizer, test_dataset, device, extractive=False, output_dir=''):
    """
    Runs inference on a test dataset and saves the results.
    
    This function processes each example in the test dataset, generates predictions
    using either sequence-to-sequence or extractive QA generation, and saves the
    results to a JSON file.
    
    Args:
        model: The model to use for inference
        tokenizer: The tokenizer for processing text
        test_dataset: The dataset to run inference on
        device: The device to run inference on (e.g., 'cuda' or 'cpu')
        extractive (bool, optional): Whether to use extractive QA generation. Defaults to False.
        output_dir (str, optional): Path to save the results. Defaults to ''.
        
    Returns:
        bool: True if inference completed successfully
    """
    output = []
    for idx, item in enumerate(tqdm(test_dataset, desc="Processing", total=len(test_dataset))):
        input = item['input']
        target = item['target']

        model_name = model.config._name_or_path
        if extractive:
            predicted = generate_output_extractive(model, tokenizer, input, device)
        else:
            predicted = generate_output(model, tokenizer, input, device)

        output.append({
            "input": input,
            "target": target,
            "predicted": predicted
        })

    with open(output_dir, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    return True