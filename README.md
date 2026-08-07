# ProvideQ Web RAG

This project retrieves scientific papers and evidence snippets for biobanking questions. It returns evidence to the downstream ProvideQ agent; it does not generate the final answer.

## Current thesis experiment

The current step compares document retrieval with:

| Query method | Description |
|---|---|
| `raw` | Original question after whitespace cleaning |
| `hyde` | One concise hypothetical biomedical passage |
| `llmexpand` | Original question plus validated biomedical synonyms and equivalent terms |

Each method can be tested with Paperclip `bm25`, `vector`, and `hybrid`. Chunking and local reranking are not used in the MRR experiment.

## Benchmark

```text
benchmark\provideq_benchmark.json
```

The benchmark contains 90 questions in five biobanking categories.

## Installation on Windows CMD

```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
set PYTHONPATH=src;.
```

Paperclip must also be installed and authenticated.

## Interweb configuration

Interweb is used through its OpenAI-compatible API. Configuration is loaded automatically from a project-local `.env` file.

Create it once:

```cmd
copy .env.example .env
notepad .env
```

Replace the placeholder API key and choose a model:

```text
INTERWEB_APIKEY=your_complete_key
WEB_RAG_LLM_MODEL=agents-a1:35b-a3b
```

The `.env` file is ignored by Git.

Check the key and model:

```cmd
python check_interweb.py --list
python check_interweb.py --test
```

The same model, temperature, prompt, and generated-query cache should be used for all ranking comparisons.

## Run the retrieval experiment

Ready-to-copy CMD commands are in:

```text
TEST_COMMANDS.txt
```

The default settings are loaded from `.env`:

```text
Questions: 90
Retrieved papers: 10
Paperclip sources: pmc,biorxiv,medrxiv,arxiv,abstracts_only
Paperclip full-corpus search: enabled
LLM provider: Interweb through OpenAI-compatible API
LLM temperature: 0
Seed: 42
```

For HyDE and LLM expansion, the evaluator automatically creates a model-specific query cache under:

```text
outputs\query_cache\
```

This ensures that vector and hybrid retrieval receive exactly the same generated query.

## Results

Each run writes:

```text
outputs\<query-method>_<paperclip-ranking>_retrieval_mrr\results.csv
```

The terminal prints:

```text
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
```

The CSV records the exact query sent to Paperclip, the retrieved paper titles and identifiers, the matched gold paper, first relevant rank, model, source settings, and hit metrics.

## Full Web RAG pipeline

```text
Question
→ Query reformulation through Interweb
→ Paperclip paper retrieval
→ Sentence-window chunking
→ Local reranking
→ Evidence selection
→ Citation-ready snippets
```

The final pipeline also loads `.env` automatically:

```python
from web_rag import run_pipeline

result = run_pipeline(
    "Is potassium stable in serum gel tubes after delayed centrifugation?",
    query_strategy="llmexpand",
)

print(result.context_text)
```
