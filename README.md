# Web_Rag

This folder contains the current Web RAG module for the ProvideQ project.

The goal of this module is to retrieve scientific evidence from biomedical literature and prepare it in a format that can later be used by the agent/LLM part of ProvideQ.

At this stage, the module does not generate the final answer. It only retrieves papers, extracts evidence snippets, ranks them, and returns an evidence context.

## Current pipeline

The current pipeline works as follows:

```text
question -> Europe PMC query -> papers -> snippets -> reranking -> evidence context
```

## Main features

* Build a Europe PMC search query from a biomedical question
* Retrieve paper metadata and abstracts
* Extract overlapping evidence snippets from abstracts
* Rank snippets using different reranking methods
* Build a citation-ready evidence context
* Save the output as JSON and text

## Reranking methods

The module currently supports three reranking methods:

* lexical
* medcpt
* hybrid

The lexical method is used as a simple baseline.
The MedCPT method uses a biomedical embedding model.
The hybrid method combines lexical and MedCPT scores.

## Run example

From inside the `Web_Rag` folder:

```bat
set PYTHONPATH=src;.
python -m web_rag.cli --question "How does delayed centrifugation affect potassium in serum samples?" --ranker lexical --page-size 8 --top-k 5 --show-query
```

MedCPT example:

```bat
set PYTHONPATH=src;.
python -m web_rag.cli --question "How does delayed centrifugation affect potassium in serum samples?" --ranker medcpt --page-size 4 --top-k 3 --medcpt-device cpu
```

Hybrid example:

```bat
set PYTHONPATH=src;.
python -m web_rag.cli --question "How does delayed centrifugation affect potassium in serum samples?" --ranker hybrid --page-size 7 --top-k 5 --medcpt-device cpu --hybrid-lexical-weight 0.45 --hybrid-medcpt-weight 0.55
```

## Output

The output contains:

* the original question
* retrieved evidence records
* source metadata such as title, PMID, DOI, year, and URL
* a context text that can later be passed to the ProvideQ agent

## Current status

This is the current Web RAG baseline implementation.
Evaluation code is not included in this folder yet because it will be reviewed and cleaned separately.
