import json
import os
import re
import string
from ultis import extract_info

# Clean the answer
def normalize_answer(text):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    # Apply all normalization steps
    text = lower(text)
    text = remove_articles(text)
    text = remove_punctuation(text)
    text = white_space_fix(text)

    return text

# calculate f1 score between the target and predicted
def calculate_f1(target, predicted):
    if target in predicted or predicted in target:
        return 1.0

    pred_normal = normalize_answer(predicted)
    tar_normal = normalize_answer(target)

    pred_tokens = pred_normal.split()
    gt_tokens = tar_normal.split()
    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(token), gt_tokens.count(token)) for token in common)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def eval_exact_math(dataset):
    em = 0.0

    for item in dataset:
        target = item['target']
        predicted = item['predicted']

        # Evaluate
        if target in predicted or predicted in target:
            em += 1.0

    return round(float(em) / len(dataset) * 100, 4)

def eval_f1_score(dataset):
    f1_score = 0.0

    for item in dataset:
        target = item['target']
        predicted = item['predicted']

        # get f1 score
        f1_item = calculate_f1(target, predicted)

        f1_score += f1_item

    return round(float(f1_score) / len(dataset) * 100, 4)

def evaluate_question_answering(dataset):
    exact_match = eval_exact_math(dataset)
    f1_score = eval_f1_score(dataset)

    return {
        "exact_match": exact_match,
        "f1_score": f1_score
    }

if __name__ == "__main__":
    print("=========== QUESTION ANSWERING ===========")
    dir = 'question_answering'

    # List to store evaluated results
    eval_res = []

    # Walk through all evaluated files
    for file in os.listdir(dir):
        # create path to tile
        path = dir + '/' + file

        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Extract model name
        _model, _meth, _task = extract_info(str(file))

        # Evaluate
        _eval = evaluate_question_answering(dataset)

        # Append to eval_res
        eval_res.append({
            "model": _model,
            "method": _meth,
            "metrics": _eval
        })

        print(f"Evaluation for {_model} of {_meth} done.")

    # Store in json file
    with open('question_answering_result.json', 'w', encoding='utf-8') as file:
        json.dump(eval_res, file, indent=4, ensure_ascii=False)
    
    print(f"======= Result stored in question_answering_result.json =======")