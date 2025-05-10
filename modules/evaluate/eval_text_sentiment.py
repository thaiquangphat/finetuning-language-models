import json
import os
from ultis import extract_info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def clean_result(target, pred):
    if pred == 'positive' or pred == 'negative':
        return pred
    if target == 'positive':
        return 'positive'
    return 'negative'

def evaluate_text_sentiment(dataset):
    targets = [item['target'] for item in dataset]
    predicteds = [clean_result(item['target'], item['predicted']) for item in dataset]

    accuracy = accuracy_score(targets, predicteds)
    precision = precision_score(targets, predicteds, pos_label='positive', average='binary')
    recall = recall_score(targets, predicteds, pos_label='positive', average='binary')
    f1 = f1_score(targets, predicteds, pos_label='positive', average='binary')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    print("=========== TEXT SENTIMENT ===========")
    dir = 'text_sentiment/inference'

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
        _eval = evaluate_text_sentiment(dataset)

        # Append to eval_res
        eval_res.append({
            "model": _model,
            "method": _meth,
            "metrics": _eval
        })

        print(f"Evaluation for {_model} of {_meth} done.")

    # Store in json file
    with open('text_sentiment/text_sentiment_result.json', 'w', encoding='utf-8') as file:
        json.dump(eval_res, file, indent=4, ensure_ascii=False)
    
    print(f"======= Result stored in text_sentiment_result.json =======")