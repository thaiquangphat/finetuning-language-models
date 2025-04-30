# Fine-tuning Results Comparison: T5-base, BART-base, GPT-2 Small

This document presents the **evaluation results** for three models: **T5-base**, **BART-base**, and **GPT-2 Small**, fine-tuned on three NLP tasks using three methods: **Full Finetuning**, **LoRA**, and **Adapters**. The results are based on the following tasks:

- **Question Answering** (SQuAD 1.0)
- **Translation** (WMT16 English-German)
- **Sentiment Classification** (IMDB)

---

## 1. **Question Answering**  
**Dataset:** [SQuAD 1.0](https://huggingface.co/datasets/rajpurkar/squad)  
**Metrics:** Exact Match (EM), F1 Score

### Results

| **Model**        || **Method**        || **Exact Match (EM)** || **F1 Score** |
|------------------||-------------------||----------------------||--------------|
| **T5-base**      || Full Finetuning   ||                      ||              |
|                 || LoRA              ||                      ||              |
|                 || Adapters          ||                      ||              |
| ^ **BART-base**  || Full Finetuning   ||                      ||              |
|                 || LoRA              ||                      ||              |
|                 || Adapters          ||                      ||              |
| ^ **GPT-2 Small**|| Full Finetuning   ||                      ||              |
|                 || LoRA              ||                      ||              |
|                 || Adapters          ||                      ||              |

---

## 2. **Translation**  
**Dataset:** [WMT16 English-German](https://huggingface.co/datasets/wmt/wmt16/tree/main/de-en)  
**Metrics:** BLEU Score

### Results

| **Model**        || **Method**        || **BLEU Score** |
|------------------||-------------------||----------------|
| **T5-base**      || Full Finetuning   ||                |
|                 || LoRA              ||                |
|                 || Adapters          ||                |
| ^ **BART-base**  || Full Finetuning   ||                |
|                 || LoRA              ||                |
|                 || Adapters          ||                |
| ^ **GPT-2 Small**|| Full Finetuning   ||                |
|                 || LoRA              ||                |
|                 || Adapters          ||                |

---

## 3. **Sentiment Classification**  
**Dataset:** [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)  
**Metrics:** Accuracy, Precision, Recall, F1 Score

### Results

| **Model**        || **Method**        || **Accuracy** || **Precision** || **Recall** || **F1 Score** |
|------------------||-------------------||--------------||---------------||------------||--------------|
| **T5-base**      || Full Finetuning   ||              ||               ||            ||              |
|                 || LoRA              ||              ||               ||            ||              |
|                 || Adapters          ||              ||               ||            ||              |
| ^ **BART-base**  || Full Finetuning   ||              ||               ||            ||              |
|                 || LoRA              ||              ||               ||            ||              |
|                 || Adapters          ||              ||               ||            ||              |
| ^ **GPT-2 Small**|| Full Finetuning   ||              ||               ||            ||              |
|                 || LoRA              ||              ||               ||            ||              |
|                 || Adapters          ||              ||               ||            ||              |

---
