# Fine-tuning Results Comparison: T5-base, BART-base, Flan-T5-small

This document presents the **evaluation results** for three models: **T5-base**, **BART-base**, and **Flan-T5-small**, fine-tuned on three NLP tasks using three methods: **Full Finetuning**, **LoRA**, and **Adapters**. The results are based on the following tasks:

- **Question Answering** (SQuAD 1.0)
- **Translation** (WMT16 English-German)
- **Sentiment Classification** (IMDB)

## How To Run
In your terminal, navigate to the `evaluate` directory and run one of the following commands to evaluate the results for a specific task:

```bash
python eval_qa.py              # For evaluating question answering results
python eval_text_sentiment.py  # For evaluating text sentiment analysis results
python eval_translation.py     # For evaluating translation results
```

Evaluation results will be stored in a file named:

```
<task>_result.json
```

**Example:**
```
question_answering_result.json
```
---

## 1. Question Answering
**Dataset:** [SQuAD 1.0](https://huggingface.co/datasets/rajpurkar/squad)  
**Metrics:** Exact Match (EM), F1 Score

### Results

<table border="1">
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
      <td>85.2312%</td>
      <td>87.3847%</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>94.3136%</td>
      <td>95.7342%</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>92.6941%</td>
      <td>94.4193%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>BART-base</strong></td>
      <td>Full Finetuning</td>
      <td>69.3850%</td>
      <td>73.6033%</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>84.1681%</td>
      <td>87.0804%</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>36.5725%</td>
      <td>40.9718%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Flan-T5-small</strong></td>
      <td>Full Finetuning</td>
      <td>82.5057%</td>
      <td>84.9403%</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>74.2509%</td>
      <td>76.9574%</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>84.1039%</td>
      <td>86.3504%</td>
    </tr>
  </tbody>
</table>

---

## 2. Translation
**Dataset:** [WMT16 English-German](https://huggingface.co/datasets/wmt/wmt16/tree/main/de-en)  
**Metrics:** BLEU Score, Cosine Similarity

### Results

<table border="1">
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center;"><strong>Model</strong></th>
      <th rowspan="2" style="text-align:center;"><strong>Method</strong></th>
      <th colspan="2" style="text-align:center;"><strong>Metrics</strong></th>
    </tr>
    <tr>
      <th>BLEU Score</th>
      <th>Cosine Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>T5-base</strong></td>
      <td>Full Finetuning</td>
      <td>0.2259</td>
      <td>0.4400</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>0.1999</td>
      <td>0.3812</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>0.2355</td>
      <td>0.4256</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>BART-base</strong></td>
      <td>Full Finetuning</td>
      <td>0.1197</td>
      <td>0.3499</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>0.0491</td>
      <td>0.2423</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>0.0001</td>
      <td>0.0642</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Flan-T5-small</strong></td>
      <td>Full Finetuning</td>
      <td>0.1383</td>
      <td>0.3559</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>0.0762</td>
      <td>0.2807</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>0.1095</td>
      <td>0.3287</td>
    </tr>
  </tbody>
</table>

---

## 3. Sentiment Classification
**Dataset:** [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)  
**Metrics:** Accuracy, Precision, Recall, F1 Score

### Results

<table border="1">
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
      <td>0.9425</td>
      <td>0.9667</td>
      <td>0.9239</td>
      <td>0.9449</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>0.7063</td>
      <td>0.7029</td>
      <td>0.7335</td>
      <td>0.7179</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>0.9084</td>
      <td>0.8946</td>
      <td>0.9298</td>
      <td>0.9119</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>BART-base</strong></td>
      <td>Full Finetuning</td>
      <td>0.9021</td>
      <td>0.8942</td>
      <td>0.9163</td>
      <td>0.9051</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>0.9184</td>
      <td>0.9274</td>
      <td>0.9112</td>
      <td>0.9192</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>0.6074</td>
      <td>0.5744</td>
      <td>0.8853</td>
      <td>0.6967</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Flan-T5-small</strong></td>
      <td>Full Finetuning</td>
      <td>0.9142</td>
      <td>0.9137</td>
      <td>0.9183</td>
      <td>0.9160</td>
    </tr>
    <tr>
      <td>LoRA</td>
      <td>0.9425</td>
      <td>0.9667</td>
      <td>0.9239</td>
      <td>0.9449</td>
    </tr>
    <tr>
      <td>Adapters</td>
      <td>0.8895</td>
      <td>0.8582</td>
      <td>0.9380</td>
      <td>0.8963</td>
    </tr>
  </tbody>
</table>
