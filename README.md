# ProvideQ Web RAG

This repository contains the Web Retrieval-Augmented Generation component developed for ProvideQ.

The system retrieves full-text scientific evidence from Paperclip, reranks the extracted evidence chunks, and returns citation-ready snippets for the downstream ProvideQ agent. It does not generate the final answer.

## Pipeline

```text
Question
→ Query reformulation
→ Paperclip full-text retrieval
→ Sentence-window chunking
→ Lexical, MedCPT, or hybrid reranking
→ Evidence selection
→ Citation-ready context
```

## Project structure

```text
provideq-web-rag/
├── benchmark/
│   └── provideq_benchmark.json
├── evaluation/
│   ├── lexical_evaluation.py
│   ├── semantic_evaluation.py
│   ├── llm_judge_evaluation.py
│   ├── run_evaluation.py
│   └── evaluation_notebook.ipynb
├── outputs/
├── src/
│   └── web_rag/
├── .gitignore
├── README.md
└── requirements.txt
```

The benchmark contains 97 questions with gold answers and source documents.

## Installation

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=src;.
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src:.
```

Paperclip must be installed and authenticated separately. The code loads the client from the Python environment or from:

```text
~/.paperclip/lib
```

Authentication can use `PAPERCLIP_API_KEY` or a stored Paperclip login.

## Run one query

```bat
python -m web_rag.cli --question "Is potassium stable in serum gel tubes if centrifugation is delayed for up to 24 hours?" --paperclip-source pmc --paperclip-ranking hybrid --limit 50 --reranker medcpt --top-k 5 --medcpt-device auto --show-query --print-context
```

Available Paperclip ranking methods:

```text
bm25
vector
hybrid
```

Available local rerankers:

```text
lexical
medcpt
hybrid
```

## Integration

The Web RAG can be called directly from another Python component:

```python
from web_rag import run_pipeline

result = run_pipeline(
    "Is potassium stable in serum gel tubes after delayed centrifugation?"
)

context_text = result.context_text
evidence_records = result.records
```

`run_pipeline()` returns the evidence and does not print it. The CLI is only a testing interface.

## Evaluation

The evaluation runner uses 20 benchmark questions by default. A value from 1 to 97 can be selected with `--num-questions`.

### Lexical evaluation

```bat
python -m evaluation.run_evaluation --evaluation lexical --num-questions 20 --paperclip-ranking hybrid --reranker lexical
```

### Semantic evaluation

```bat
python -m evaluation.run_evaluation --evaluation semantic --num-questions 20 --paperclip-ranking hybrid --reranker medcpt --semantic-device auto
```

### LLM-as-a-Judge

The default judge uses Ollama with `qwen2.5:7b-instruct`.

```bat
ollama pull qwen2.5:7b-instruct
python -m evaluation.run_evaluation --evaluation judge --num-questions 20 --paperclip-ranking hybrid --reranker hybrid
```

To run all three layers:

```bat
python -m evaluation.run_evaluation --evaluation all --num-questions 20 --paperclip-ranking hybrid --reranker hybrid
```

The same random seed selects the same benchmark questions across experiments. The default seed is `42`.

## Evaluation outputs

Results are saved under:

```text
outputs/<paperclip-ranking>_<reranker>_<evaluation>/results.csv
```

Each CSV contains:

```text
question_id, question, answers, gold_answer, score
```

## Notebook

Open:

```text
evaluation/evaluation_notebook.ipynb
```

The notebook reads evaluation CSV files from `outputs/` and compares retrieval and reranking configurations for the three evaluation layers.
