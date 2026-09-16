# Benchmark testing

Run commands from Windows CMD in the repository root with `.venv`
activated. Keep the benchmark, seed, candidate limits, and retrieval limit fixed
while changing one factor at a time. Generated files stay under `outputs/` and
are intentionally ignored by Git.

## 1. Preflight

~~~cmd
python -m unittest discover -s tests -v
python -c "from web_rag.config import get_settings; s=get_settings(); print(s.query_strategy, s.paperclip_ranking, s.retrieval_limit, s.reranker, s.top_k, s.max_chunks_per_paper)"
~~~

The fixed baseline is `raw hybrid 20 medcpt 20 4`.

Run one end-to-end request before any benchmark sweep:

~~~cmd
python -m web_rag.cli "Is potassium stable in serum gel tubes after delayed centrifugation?" --output-dir outputs\smoke --show-info
~~~

Accept the smoke test only when full text was loaded, 20 distinct evidence
records were returned, no chunk exceeds 512 tokens, and every record has source,
paper-rank, rerank-rank, section, and token-count metadata.

## 2. The 15 document-retrieval tests

The sweep compares five retrieval settings with three query preparations. Every
test returns at most 20 papers and stops before full-text loading, chunking, and
reranking.

| Retrieval setting | Raw | Anchored HyDE | Anchored LLM expansion |
| --- | --- | --- | --- |
| Paperclip BM25 | 01 | 02 | 03 |
| Paperclip vector | 04 | 05 | 06 |
| Paperclip hybrid | 07 | 08 | 09 |
| Europe PMC with synonyms | 10 | 11 | 12 |
| Europe PMC without synonyms | 13 | 14 | 15 |

Both Europe PMC settings use the same `multi` mode, so synonym expansion is the
only retrieval difference between them. For Paperclip, anchored queries fuse the
raw and reformulated rankings. For Europe PMC, the reformulated query itself
starts with the unchanged raw question.

Check the exact test names without sending any retrieval requests:

~~~cmd
python -m evaluation.run_retrieval_sweep --list-configs
~~~

## 3. Run the five-question pilot

Start with the same five benchmark questions for all 15 configurations:

~~~cmd
python -m evaluation.run_retrieval_sweep --num-questions 5 --seed 42 --no-resume
~~~

The files are written under `outputs\retrieval_15_tests`. Each numbered test
folder contains its own `results.csv`, and the root contains `summary.csv`:

~~~text
outputs\retrieval_15_tests\
  01_paperclip_bm25_raw\results.csv
  02_paperclip_bm25_anchored_hyde\results.csv
  ...
  15_europepmc_no_synonyms_anchored_llm_expansion\results.csv
  summary.csv
~~~

Generated HyDE and expansion queries are cached once and reused across retrieval
settings, which keeps the query preparation identical for a fair comparison.
Use a new output directory, or rerun with `--no-resume`, after deliberately
changing a query prompt or model. The prompt version is part of the query-cache
name, so older generated queries cannot be reused silently.

## 4. Validate and expand the run

Before using the full benchmark, inspect every `results.csv` for errors and check
that the raw question remains visible at the start of every anchored query. Use
Recall@20 as the primary document-retrieval measure, followed by Recall@10,
MRR@10, and MRR@20. Both MRR values are included in `summary.csv` and the console
comparison.

If a transient service or connection problem affects only some saved rows, retry
those rows without repeating completed requests:

~~~cmd
python -m evaluation.run_retrieval_sweep --num-questions 5 --seed 42 --retry-errors
~~~

The Europe PMC error columns record the full query-variant list, the exact failed
variant, and its index. A multi-query configuration remains an error if any of its
variants fails, so partial source responses cannot make configurations
incomparable.

After the pilot is clean, run all 90 benchmark questions:

~~~cmd
python -m evaluation.run_retrieval_sweep --num-questions 90 --seed 42 --no-resume
~~~

If an interrupted run used the same settings, omit `--no-resume` to continue
from its saved rows. Do not compare results created with different question
counts, seeds, retrieval limits, prompts, or models.

## 5. Final 20-chunk evaluation

Evaluate the unchanged default pipeline first on one known question, then five,
then all 90:

~~~cmd
python -m evaluation.run_chunk_evaluation --question-id Q039 --output outputs\chunk_evaluation\q039\results.json --no-resume
python -m evaluation.run_chunk_evaluation --num-questions 5 --seed 42 --output outputs\chunk_evaluation\five\results.json --no-resume
python -m evaluation.run_chunk_evaluation --num-questions 90 --output outputs\chunk_evaluation\full\results.json --no-resume
~~~

Each row retains all returned chunks and metadata plus the best lexical and
semantic score and rank. Use those saved chunks for the later LLM-as-a-judge
review. A configuration is not ready to become the new default until its run has
no unexplained errors, its settings are recorded, and spot-checks show no
keyword-only, bibliography, acknowledgement, or duplicate evidence fragments.

## 6. Claude-controlled Paperclip pilot

Run this only from the experimental branch. It is an isolated document-retrieval
comparison and does not modify the fixed pipeline.

Install the optional Agent SDK:

~~~cmd
python -m pip install -e ".[claude]"
~~~

Add `ANTHROPIC_API_KEY` and `PAPERCLIP_API_KEY` to the local `.env`. The second
key is created at <https://paperclip.gxl.ai/keys>. Do not commit either key.

Run the single-question comparison:

~~~cmd
python -m evaluation.run_claude_paperclip_test --question-id Q003
~~~

The fixed side uses the raw question, Paperclip hybrid ranking, the configured
academic sources, and a final limit of 20 papers. The Claude side has the same
source, ranking, and 20-paper limit, but may issue at most three Paperclip MCP
searches. It is instructed not to use `map`, `reduce`, routines, general web
search, or final-answer generation.

Inspect both files before interpreting the printed label:

~~~text
outputs\claude_paperclip_smoke\Q003\comparison.json
outputs\claude_paperclip_smoke\Q003\papers.csv
~~~

`comparison.json` records the adaptive queries, MCP tool calls, both ranked
paper lists, gold match and first-relevant rank, token usage, cost when reported,
runtime, and the raw structured response. `papers.csv` provides a direct
paper-by-paper review. One question is only a smoke test; an `improved` label is
not enough to replace the default without a larger repeated benchmark.
