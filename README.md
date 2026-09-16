# ProvideQ Web RAG

This module retrieves scientific papers and returns evidence chunks to the
downstream ProvideQ agent. It does not generate the final answer.

## Default pipeline

The no-argument defaults represent the best fixed pipeline to test first:

| Stage | Default |
|---|---|
| Query | Raw question |
| Paper retrieval | Paperclip hybrid, full corpus |
| Paper retrieval limit | 20 papers |
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
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
curl.exe -fL "https://paperclip.gxl.ai/paperclip.whl" -o "%TEMP%\gxl_paperclip-0.7.49-py3-none-any.whl"
python -m pip install "%TEMP%\gxl_paperclip-0.7.49-py3-none-any.whl"
python -m pip install -e .
python -m pip check
copy .env.example .env
~~~

The explicit download filename is required because the filename supplied by the
Paperclip endpoint is not a valid Python wheel filename on Windows. Verify that
Paperclip 0.7.49 was installed in the active environment:

~~~cmd
python -c "from importlib.metadata import version; v=version('gxl-paperclip'); print(v); assert v == '0.7.49'"
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
python -c "from web_rag.config import get_settings; s=get_settings(); print('retrieval_limit:',s.retrieval_limit); print('reranker:',s.reranker); print('top_k:',s.top_k); print('max_chunks_per_paper:',s.max_chunks_per_paper)"
~~~

The fixed default should print `20`, `medcpt`, `20`, and `4`. If it does not,
remove the old lines from `.env` or set them to:

~~~dotenv
WEB_RAG_RETRIEVAL_LIMIT=20
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

The stable integration boundary is run_pipeline(question). Raw queries, a
20-paper Paperclip hybrid retrieval limit, token-aware chunking, MedCPT
cross-encoder reranking, and top-20 evidence selection are all defaults.

## Later retrieval experiments

The document-retrieval sweep compares exactly 15 configurations: raw, anchored
HyDE, and anchored LLM expansion across Paperclip BM25, vector, and hybrid, plus
Europe PMC multi-query retrieval with and without synonyms. Each configuration
gets a clearly numbered folder and `results.csv`; the sweep also writes a shared
`summary.csv`. These tests stop before chunking and reranking, so they do not
alter the fixed default pipeline. Follow [TESTING.md](TESTING.md) for the pilot
and full-benchmark commands.

## Optional Claude + Paperclip experiment

The `experiment/claude-paperclip` branch contains a separate document-retrieval
test. Claude can make up to three adaptive searches through Paperclip's hosted
MCP server, while the comparison baseline sends the unchanged raw question to
Paperclip hybrid retrieval. Neither path performs chunking or answer generation
in this first comparison.

The experiment does not change `run_pipeline`, its defaults, or Aryan's
20-chunk integration contract. It also explicitly ignores `.claude` project and
user customizations so that the checked-in prompt is the only agent instruction.

Install the optional dependency and add two private values to `.env`:

~~~cmd
python -m pip install -e ".[claude]"
~~~

~~~dotenv
ANTHROPIC_API_KEY=your_anthropic_api_key
PAPERCLIP_API_KEY=your_paperclip_api_key
~~~

Create the Paperclip key at <https://paperclip.gxl.ai/keys>. Run the default
one-question comparison with:

~~~cmd
python -m evaluation.run_claude_paperclip_test --question-id Q003
~~~

Q003 is intentional: the saved raw-hybrid run missed its gold paper, while an
anchored expansion found it, so it provides room for an adaptive method to help.
The command writes `comparison.json` and `papers.csv` under
`outputs\claude_paperclip_smoke\Q003`. A result is labelled `improved` only when
Claude moves the first matching gold paper to a better rank than the fixed
baseline, or finds it when the baseline does not.
