# ProvideQ Web RAG

A modular Web RAG project for biomedical fallback retrieval in ProvideQ.

The goal is to create a transparent, thesis-ready biomedical Web RAG pipeline that can retrieve external scientific evidence when the internal ProvideQ knowledge sources do not contain enough information.

This repository is being built incrementally. Each upgrade adds one clear capability to the pipeline.

## Project goal

ProvideQ is a curated biomedical database for pre-analytical variability. The Web RAG component built in this repository focuses on external scientific fallback retrieval.

The module should eventually:

- receive a biomedical stability question
- build a suitable scientific search query
- retrieve papers from trusted biomedical sources
- normalize metadata
- extract evidence snippets
- rank evidence snippets
- generate grounded answers with citations
- support reproducible experiments and evaluation


## Current pipeline

The current implementation supports:

```text
question
   ↓
query building
   ↓
Europe PMC search
   ↓
paper metadata normalization
   ↓
evidence snippet extraction

``` 
## Current project structure
```
provideq-web-rag/
│
├── src/
│   └── web_rag/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── query_builder.py
│       ├── snippet_extractor.py
│       ├── source_client.py
│       └── text_utils.py
│
├── scripts/
├── tests/
├── outputs/
├── .gitignore
├── README.md
└── requirements.txt

```