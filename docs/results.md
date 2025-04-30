# Fine-tuning Results Comparison: T5-base, BART-base, GPT-2 Small

This document presents the **evaluation results** for three models: **T5-base**, **BART-base**, and **GPT-2 Small**, fine-tuned on three NLP tasks using three methods: **Full Finetuning**, **LoRA**, and **Adapters**. The results are based on the following tasks:

- **Question Answering** (SQuAD 1.0)
- **Translation** (WMT16 English-German)
- **Sentiment Classification** (IMDB)

---

## 1. Question Answering
**Dataset:** [SQuAD 1.0](https://huggingface.co/datasets/rajpurkar/squad)  
**Metrics:** Exact Match (EM), F1 Score

### Results

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center;"><strong>Model</strong></th>
      <th rowspan="2" style="text-align:center;"><strong>Method</strong></th>
      <th colspan="2" style="text-align:center;"><strong>Metrics</strong></th>
    </tr>
    <tr>
      <th><strong>Exact Match (EM)</strong></th>
      <th><strong>F1 Score</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>T5-base</strong></td>
      <td>Full Finetuning</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><strong>BART-base</strong></td>
      <td>Full Finetuning</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><strong>GPT-2 Small</strong></td>
      <td>Full Finetuning</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

---

## 2. Translation
**Dataset:** [WMT16 English-German](https://huggingface.co/datasets/wmt/wmt16/tree/main/de-en)  
**Metrics:** BLEU Score

### Results

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center;"><strong>Model</strong></th>
      <th rowspan="2" style="text-align:center;"><strong>Method</strong></th>
      <th colspan="2" style="text-align:center;"><strong>Metrics</strong></th>
    </tr>
    <tr>
      <th>BLEU Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>T5-base</strong></td>
      <td>Full Finetuning</td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><strong>BART-base</strong></td>
      <td>Full Finetuning</td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><strong>GPT-2 Small</strong></td>
      <td>Full Finetuning</td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
    </tr>
  </tbody>
</table>

---

## 3. Sentiment Classification
**Dataset:** [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)  
**Metrics:** Accuracy, Precision, Recall, F1 Score

### Results

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center;"><strong>Model</strong></th>
      <th rowspan="2" style="text-align:center;"><strong>Method</strong></th>
      <th colspan="4" style="text-align:center;"><strong>Metrics</strong></th>
    </tr>
    <tr>
      <th><strong>Accuracy</strong></th>
      <th><strong>Precision</strong></th>
      <th><strong>Recall</strong></th>
      <th><strong>F1 Score</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>T5-base</strong></td>
      <td>Full Finetuning</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><strong>BART-base</strong></td>
      <td>Full Finetuning</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><strong>GPT-2 Small</strong></td>
      <td>Full Finetuning</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
