# ProvideQ Web RAG

This module retrieves scientific papers and returns evidence chunks to the
downstream ProvideQ agent. It does not generate the final answer.

## Default pipeline

The no-argument defaults represent the best fixed pipeline to test first:

| Stage | Default |
|---|---|
| Query | Raw question |
| Paper retrieval | Paperclip hybrid, full corpus |
| Retrieved papers | 10 |
| Article text | Full text requested explicitly |
| Chunking | Section-preserving, tokenizer-aware |
| Chunk size | Maximum 512 MedCPT tokens |
| Overlap | 20%, using at most five complete trailing sentences |
| Chunk reranker | ncbi/MedCPT-Cross-Encoder |
| Returned evidence | 20 distinct chunks when at least 20 are available |

Adjacent sentences from the same section are merged until the token budget is
reached. Chunks never cross detected section boundaries. References,
acknowledgements, funding, indexing-keyword lists, and similar non-evidence
content are skipped. Repeated findings from an abstract and the body of the same
paper are removed before the final selection.

This adapts Aryan's Docling HybridChunker behavior to the plain article text
returned by Paperclip. It deliberately does not add Docling as another parsing
layer because Paperclip has already extracted the text.

## Solr compatibility

Aryan uses Solr to store and retrieve candidates from his persistent local
corpus. Web RAG has no persistent corpus: Paperclip already performs its
first-stage hybrid retrieval. Therefore Solr is not duplicated here.

The compatible hand-off point is the output of run_pipeline: up to 20
MedCPT-reranked chunks with source, paper rank, rerank rank, section, sentence
range, token count, full-text status, and score metadata. Aryan's agent can
consume these records directly.

## Windows setup

Use Python 3.12 and the tested Paperclip environment:

~~~cmd
py -3.12 -m venv .venv312
call .venv312\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
copy .env.example .env
~~~

Paperclip 0.7.49 must be installed and authenticated separately. Verify it:

~~~cmd
python -c "from importlib.metadata import version; print(version('gxl-paperclip'))"
~~~

Add your own API key to `.env` before testing HyDE or LLM expansion. The raw
default pipeline does not call the LLM. The first chunking/reranking run
downloads ncbi/MedCPT-Cross-Encoder from Hugging Face. The semantic evaluator
separately downloads BAAI/bge-m3.

## Verify the active defaults

Local `.env` entries and Windows environment variables beginning with
`WEB_RAG_` override the defaults in the code. This is useful for experiments,
but settings from an older run can silently change the smoke test. Check the
effective values before testing:

~~~cmd
python -c "from web_rag.config import get_settings; s=get_settings(); print('reranker:',s.reranker); print('top_k:',s.top_k); print('max_chunks_per_paper:',s.max_chunks_per_paper)"
~~~

The fixed default should print `medcpt`, `20`, and `4`. If it does not,
remove the old lines from `.env` or set them to:

~~~dotenv
WEB_RAG_RERANKER=medcpt
WEB_RAG_TOP_K=20
WEB_RAG_MAX_CHUNKS_PER_PAPER=4
~~~

## Run one end-to-end smoke test

~~~cmd
python -m web_rag.cli "Is potassium stable in serum gel tubes after delayed centrifugation?" --output-dir outputs\smoke --show-info
~~~

The command writes:

~~~text
outputs\smoke\evidence.json
outputs\smoke\context.txt
~~~

Check full text, chunk count, token limits, and metadata:

~~~cmd
python -c "import json; d=json.load(open(r'outputs\smoke\evidence.json',encoding='utf-8')); print('full-text papers:',d['pipeline']['full_text_papers_count']); print('chunks:',len(d['records'])); print('max tokens:',max((x['token_count'] for x in d['records']),default=0)); [print(x['citation_id'],x['source']['paper_id'],x['paper_retrieval_rank'],x['rerank_rank'],x['section'],x['token_count'],x['has_full_text']) for x in d['records']]"
~~~

Expected for a normal result with enough evidence:

- full-text papers is greater than zero;
- 20 records are returned;
- every token count is at most 512;
- every record has source and ranking metadata.

The pipeline returns fewer than 20 only when fewer than 20 sufficiently distinct
non-empty chunks are available. It never creates artificial duplicates.

## Run the unit tests

The tests do not download models or contact Paperclip:

~~~cmd
python -m unittest discover -s tests -v
~~~

## Evaluate the final chunks

Start with one known benchmark question:

~~~cmd
python -m evaluation.run_chunk_evaluation --question-id Q039 --no-resume
~~~

Then run a five-question check:

~~~cmd
python -m evaluation.run_chunk_evaluation --num-questions 5 --seed 42 --no-resume
~~~

After that, run all 90 questions:

~~~cmd
python -m evaluation.run_chunk_evaluation --num-questions 90 --no-resume
~~~

The evaluator saves incrementally to:

~~~text
outputs\default_chunk_evaluation\results.json
outputs\default_chunk_evaluation\summary.json
~~~

results.json contains all 20 chunks and their metadata for each question,
together with the best lexical and semantic match scores and ranks. This file
can be used for the later LLM-as-a-judge review.

If GPU auto-detection causes a problem, add:

~~~cmd
--device cpu
~~~

## Python integration

~~~python
from web_rag import run_pipeline

result = run_pipeline(
    "Is potassium stable in serum gel tubes after delayed centrifugation?"
)

for chunk in result.records:
    print(
        chunk.citation_id,
        chunk.source.paper_id,
        chunk.rerank_rank,
        chunk.score,
        chunk.evidence_text,
    )
~~~

The stable integration boundary is run_pipeline(question). Raw queries,
Paperclip hybrid retrieval, token-aware chunking, MedCPT cross-encoder
reranking, and top-20 selection are all defaults.

## Later retrieval experiments

The previous document-retrieval experiments remain available under evaluation.
Raw, HyDE, and LLM expansion can still be compared with Paperclip BM25, vector,
or hybrid ranking. They do not alter the fixed default pipeline unless selected
explicitly. Follow [TESTING.md](TESTING.md) to run the comparisons in stages
with matching benchmark subsets and seeds.
