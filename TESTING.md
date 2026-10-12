# Benchmark testing

Run commands from Windows CMD in the repository root with `.venv`
activated. Keep the benchmark, seed, candidate limits, and retrieval limit fixed
while changing one factor at a time. Results under `outputs/` are tracked by Git;
new query and Europe PMC caches are ignored. The approved query caches from the
original retrieval sweep remain tracked as inputs to the fusion experiments.

## 1. Preflight

~~~cmd
python -m unittest discover -s tests -v
python -c "from web_rag.config import get_settings; s=get_settings(); print(s.retrieval_system, s.query_strategy, s.paperclip_ranking, s.retrieval_limit, s.reranker, s.top_k, s.max_chunks_per_paper)"
~~~

The selected default is `fusion llmexpand vector 20 medcpt 20 4`.

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
Use a new output directory after changing a query prompt or model. Increment the
prompt version when editing a generation prompt; it is part of the query-cache
name. `--no-resume` reruns retrieval but does not remove a reusable query cache.

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

## 5. Focused retrieval-fusion tests

The completed 15-test benchmark identified Paperclip vector as the strongest
single Paperclip ranking and raw Europe PMC without synonyms as the most useful
independent lexical source. The fusion sweep compares vector against hybrid
under the same three query preparations while keeping Europe PMC fixed:

| Test | Paperclip side | Europe PMC side | Final fusion |
| --- | --- | --- | --- |
| 01 | Vector, raw | Multi-query, raw, no synonyms | Equal-weight RRF |
| 02 | Vector, raw + anchored HyDE | Multi-query, raw, no synonyms | Equal-weight RRF |
| 03 | Vector, raw + anchored LLM expansion | Multi-query, raw, no synonyms | Equal-weight RRF |
| 04 | Hybrid, raw | Multi-query, raw, no synonyms | Equal-weight RRF |
| 05 | Hybrid, raw + anchored HyDE | Multi-query, raw, no synonyms | Equal-weight RRF |
| 06 | Hybrid, raw + anchored LLM expansion | Multi-query, raw, no synonyms | Equal-weight RRF |

Each source contributes up to 30 candidates. Duplicate papers are merged by
identifier, equal-weight RRF uses `k=60`, and only the best 20 fused papers are
evaluated. Europe PMC always receives the raw question; HyDE and LLM expansion
are anchored to the raw Paperclip result with the previously tested `2:1`
raw-to-reformulated weighting.

Review the exact configurations without making requests:

~~~cmd
python -m evaluation.run_fusion_sweep --list-configs
~~~

Run a five-question smoke test first:

~~~cmd
python -m evaluation.run_fusion_sweep --num-questions 5 --seed 42 --output-dir outputs\retrieval_fusion_90 --no-resume
~~~

If all six folders contain five successful rows with no unexplained warning,
run the fixed 90-question comparison:

~~~cmd
python -m evaluation.run_fusion_sweep --num-questions 90 --seed 42 --output-dir outputs\retrieval_fusion_90
~~~

The sweep reuses the approved HyDE and LLM-expansion query caches from the v6
90-question run. This keeps query preparation identical and avoids new LLM
generation. Results are written to six clearly numbered folders plus
`summary.csv`. The 90-question command resumes the five clean pilot rows and
reuses the Europe PMC response cache. If retrieval is interrupted, run the same
command again; use `--retry-errors` only for rows affected by transient service
or connection errors.

Do not change source weights or RRF constants during this comparison. The
selected default is test 03: Paperclip vector raw/anchored-LLM-expansion fusion
plus raw Europe PMC without synonyms.

## 6. Final 20-chunk evaluation

Evaluate the selected default pipeline first on one known question, then five,
then all 90:

~~~cmd
python -m evaluation.run_chunk_evaluation --question-id Q039 --output outputs\chunk_evaluation\q039\results.json --no-resume
python -m evaluation.run_chunk_evaluation --num-questions 5 --seed 42 --output outputs\chunk_evaluation\five\results.json --no-resume
python -m evaluation.run_chunk_evaluation --num-questions 90 --output outputs\chunk_evaluation\full\results.json --no-resume
~~~

Each row retains the returned chunks, metadata, and best lexical and semantic
match for each gold nugget. These saved chunks are the input to the separate
LLM-as-a-judge review. Check the source warnings and effective settings before
comparing runs.

To compare all three passage re-rankers on 90 questions without overwriting the
saved thesis results:

~~~cmd
python -m evaluation.run_chunk_evaluation --num-questions 90 --reranker medcpt --output outputs\reranker_comparison\medcpt\results.json --no-resume
python -m evaluation.run_chunk_evaluation --num-questions 90 --reranker lexical --output outputs\reranker_comparison\lexical\results.json --no-resume
python -m evaluation.run_chunk_evaluation --num-questions 90 --reranker hybrid --output outputs\reranker_comparison\hybrid\results.json --no-resume
~~~

Each command runs document retrieval and chunking before re-ranking; it does not
re-rank the previous command's chunks. Europe PMC responses may be reused from
the cache, while live candidate lists can change between runs.

Chunk evaluation stops on a retrieval-source failure by default. Repeat the same
command with `--retry-errors` and without `--no-resume` to retry failed questions
and retain completed rows.
