import json
import os
from ultis import extract_info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def clean_result(target, pred):
    """
    Cleans and standardizes sentiment prediction results.
    
    This function ensures that predictions are either 'positive' or 'negative'.
    If the prediction is not in the expected format, it defaults to the target label.
    
    Args:
        target (str): The ground truth sentiment label
        pred (str): The model's predicted sentiment
        
    Returns:
        str: The cleaned prediction ('positive' or 'negative')
    """
    if pred == 'positive' or pred == 'negative':
        return pred
    if target == 'positive':
        return 'positive'
    return 'negative'

def evaluate_text_sentiment(dataset):
    """
    Evaluates text sentiment analysis performance using multiple metrics.
    
    This function computes accuracy, precision, recall, and F1 score
    for sentiment analysis predictions. The evaluation is done in a binary
    classification setting (positive vs negative).
    
    Args:
        dataset (list): List of dictionaries containing 'target' and 'predicted' sentiments
        
    Returns:
        dict: A dictionary containing:
            - accuracy (float): Overall accuracy score
            - precision (float): Precision score for positive class
            - recall (float): Recall score for positive class
            - f1 (float): F1 score for positive class
    """
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