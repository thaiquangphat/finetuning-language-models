import json
import os
import re
import string
from ultis import extract_info
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalize_text(text):
    """
    Normalizes text for translation evaluation.
    
    This function performs basic text normalization by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Stripping whitespace
    
    Args:
        text (str): The input text to normalize
        
    Returns:
        str: The normalized text
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.strip()
    return text

def evaluate_translation(dataset):
    """
    Evaluates machine translation performance using BLEU score and cosine similarity.
    
    This function computes two metrics:
    1. BLEU score: Measures n-gram overlap between predicted and reference translations
    2. Cosine similarity: Measures semantic similarity using TF-IDF vectors
    
    Args:
        dataset (list): List of dictionaries containing 'target' and 'predicted' translations
        
    Returns:
        dict: A dictionary containing:
            - bleu (float): BLEU score between 0 and 1
            - cosine_similarity (float): Cosine similarity score between 0 and 1
    """
    smooth = SmoothingFunction().method4

    references = []
    hypotheses = []
    similarities = []

    for item in dataset:
        target = normalize_text(item['target'])
        predicted = normalize_text(item['predicted'])

        ref_tokens = target.split()
        pred_tokens = predicted.split()

        references.append([ref_tokens])
        hypotheses.append(pred_tokens)

        try:
            vectorizer = TfidfVectorizer(stop_words=None).fit([target, predicted])
            tfidf_matrix = vectorizer.transform([target, predicted])
            sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            similarities.append(sim)
        except ValueError:
            continue

    bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    similarity = sum(similarities) / len(similarities)

    return {
        "bleu": bleu,
        "cosine_similarity": similarity
    }

if __name__ == "__main__":
    print("=========== TRANSLATION ===========")
    dir = 'translation/inference'

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
        _eval = evaluate_translation(dataset)

        # Append to eval_res
        eval_res.append({
            "model": _model,
            "method": _meth,
            "metrics": _eval
        })

        print(f"Evaluation for {_model} of {_meth} done.")

    # Store in json file
    with open('translation/translation_result.json', 'w', encoding='utf-8') as file:
        json.dump(eval_res, file, indent=4, ensure_ascii=False)
    
    print(f"======= Result stored in translation_result.json =======")
