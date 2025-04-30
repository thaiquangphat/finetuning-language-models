import json
import torch
from tqdm import tqdm

def generate_output(model, tokenizer, input, device, max_length=512):
    input_text = input
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_length=64, num_beams=4, early_stopping=True) #if not peft ==> generate(input, max_length=64, ...)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def generate_output_extractive(model, tokenizer, input, device, max_length=512):
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

def run_prediction(model, tokenizer, test_dataset, device, output_dir=''):
    output = []
    for idx, item in enumerate(tqdm(test_dataset, desc="Processing", total=len(test_dataset))):
        input = item['input']
        target = item['target']

        model_name = model.config._name_or_path
        if 'bart' in model_name:
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