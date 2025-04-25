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

def run_prediction(model, tokenizer, test_dataset, device, output_dir=''):
    output = []
    for idx, item in enumerate(tqdm(test_dataset, desc="Processing", total=len(test_dataset))):
        input = item['input']
        target = item['target']

        predicted = generate_output(model, tokenizer, input, device)

        output.append({
            "input": input,
            "target": target,
            "predicted": predicted
        })

    with open(output_dir, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    return True
