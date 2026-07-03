# Web RAG for ProvideQ

This project contains the Web Retrieval-Augmented Generation module developed for ProvideQ.

The module retrieves scientific evidence from biomedical literature and prepares citation-ready evidence that can later be used by the ProvideQ agent or LLM.

The Web RAG module does not generate the final answer.

## Pipeline

```text
Question
→ Query reformulation
→ Europe PMC retrieval
→ Paper normalization
→ Snippet extraction
→ Reranking
→ Evidence context
```

## Features

* Biomedical query reformulation for Europe PMC
* Scientific paper and abstract retrieval
* Paper metadata normalization
* Overlapping sentence-window snippet extraction
* Multiple reranking methods
* Evidence deduplication and filtering
* Citation-ready evidence context
* JSON and text serialization
* Retrieval evaluation on a benchmark dataset

## Reranking Methods

The following reranking methods are available:

* `lexical` — BM25-based lexical relevance scoring
* `medcpt` — biomedical semantic ranking using MedCPT embeddings
* `hybrid` — weighted combination of lexical and MedCPT scores

## Project Structure

```text
Web_Rag/
├── benchmarks/
├── eval/
├── notebooks/
├── outputs/
├── src/
│   └── web_rag/
├── requirements.txt
└── README.md
```

The main retrieval implementation is located in:

```text
src/web_rag/
```

The evaluation implementation is located in:

```text
eval/
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set the Python path before running the module:

```bat
set PYTHONPATH=src;.
```

## Run the Web RAG Pipeline

### Lexical reranker

```bat
python -m web_rag.cli --question "How does delayed centrifugation affect potassium in serum samples?" --ranker lexical --page-size 8 --top-k 5 --show-query
```

### MedCPT reranker

```bat
python -m web_rag.cli --question "How does delayed centrifugation affect potassium in serum samples?" --ranker medcpt --page-size 8 --top-k 5 --medcpt-device cpu --show-query
```

### Hybrid reranker

```bat
python -m web_rag.cli --question "How does delayed centrifugation affect potassium in serum samples?" --ranker hybrid --page-size 8 --top-k 5 --medcpt-device cpu --hybrid-lexical-weight 0.45 --hybrid-medcpt-weight 0.55 --show-query
```

## Retrieval Output

The retrieval output contains:

* Original question
* Reformulated Europe PMC query
* Ranked evidence snippets
* Ranking scores
* Paper title
* PMID
* DOI
* Publication year
* Authors
* Source URL
* Citation-ready context text

## Evaluation

The benchmark contains:

* `question_id`
* `question`
* `gold_answer`
* `gold_nuggets`

The current evaluation framework supports two layers.

### Lexical Evaluation

* `ROUGE1_Nugget@k`
* `ROUGEL_Nugget@k`
* `ROUGE_Nugget@k`
* `BM25_Nugget@k`

### Semantic Evaluation

* `SemanticNuggetMatch@k`
* `SemanticAnswerMatch@k`

## Run Retrieval on the Benchmark

Example using the lexical reranker:

```bat
python -m eval.run_evaluation --run-retrieval --benchmark benchmarks\provideq_web_rag_evidence_benchmark_20.csv --ranker lexical --page-size 8 --top-k 5
```

The retrieval results are saved to:

```text
outputs/ranker_lexical/retrieval_results.csv
```

The same command can be run with:

```text
--ranker medcpt
--ranker hybrid
```

## Run Lexical Evaluation

```bat
python -m eval.run_evaluation --run-lexical --benchmark benchmarks\provideq_web_rag_evidence_benchmark_20.csv --ranker lexical --k 5 --lexical-metrics all
```

Results are saved to:

```text
outputs/evaluation/lexical/evaluation_results.csv
```

## Run Semantic Evaluation

```bat
python -m eval.run_evaluation --run-semantic --benchmark benchmarks\provideq_web_rag_evidence_benchmark_20.csv --ranker lexical --k 5 --semantic-metrics all --semantic-device cpu
```

Results are saved to:

```text
outputs/evaluation/semantic/evaluation_results.csv
```

## Evaluation Notebook

The comparison notebook is located at:

```text
notebooks/evaluation_comparison_notebook.ipynb
```

It compares the lexical, MedCPT, and hybrid rerankers across the implemented lexical and semantic evaluation metrics.
