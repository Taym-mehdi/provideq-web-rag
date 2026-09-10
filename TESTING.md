# Benchmark testing

Run commands from Windows CMD in the repository root with `.venv312`
activated. Keep the benchmark, seed, candidate limits, and retrieval limit fixed
while changing one factor at a time. Generated files stay under `outputs/` and
are intentionally ignored by Git.

## 1. Preflight

~~~cmd
python -m unittest discover -s tests -v
python -c "from web_rag.config import get_settings; s=get_settings(); print(s.query_strategy, s.paperclip_ranking, s.retrieval_limit, s.reranker, s.top_k, s.max_chunks_per_paper)"
~~~

The fixed baseline is `raw hybrid 10 medcpt 20 4`.

Run one end-to-end request before any benchmark sweep:

~~~cmd
python -m web_rag.cli "Is potassium stable in serum gel tubes after delayed centrifugation?" --output-dir outputs\smoke --show-info
~~~

Accept the smoke test only when full text was loaded, 20 distinct evidence
records were returned, no chunk exceeds 512 tokens, and every record has source,
paper-rank, rerank-rank, section, and token-count metadata.

## 2. Paperclip ranking comparison

Start with five questions. These runs vary only Paperclip ranking:

~~~cmd
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever paperclip --query-strategy raw --paperclip-ranking bm25 --paperclip-candidate-limit 30 --retrieval-limit 10 --output-dir outputs\document_retrieval --no-resume
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever paperclip --query-strategy raw --paperclip-ranking vector --paperclip-candidate-limit 30 --retrieval-limit 10 --output-dir outputs\document_retrieval --no-resume
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever paperclip --query-strategy raw --paperclip-ranking hybrid --paperclip-candidate-limit 30 --retrieval-limit 10 --output-dir outputs\document_retrieval --no-resume
~~~

Repeat the same commands with `--num-questions 90` only after the five-question
check completes without errors.

## 3. Query preparation comparison

Keep Paperclip hybrid fixed. Raw is the control above. HyDE and controlled LLM
expansion retain the raw question as an anchor by fusing raw and reformulated
Paperclip rankings:

~~~cmd
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever paperclip --query-strategy hyde --paperclip-ranking hybrid --paperclip-query-fusion --query-fusion-rrf-k 10 --reformulated-query-weight 0.5 --paperclip-candidate-limit 30 --retrieval-limit 10 --output-dir outputs\document_retrieval --no-resume
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever paperclip --query-strategy llmexpand --paperclip-ranking hybrid --paperclip-query-fusion --query-fusion-rrf-k 10 --reformulated-query-weight 0.5 --paperclip-candidate-limit 30 --retrieval-limit 10 --output-dir outputs\document_retrieval --no-resume
~~~

Do not reuse an old `--llm-cache` across prompt versions. If a temporary service
failure produced warnings, resume with `--resume --retry-warnings`.

## 4. Other retrieval sources

After selecting the best Paperclip and query settings, compare Europe PMC and
source fusion on the same questions:

~~~cmd
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever europepmc --query-strategy raw --europepmc-mode direct --europepmc-synonym --europepmc-candidate-limit 30 --retrieval-limit 10 --output-dir outputs\document_retrieval --no-resume
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever europepmc --query-strategy raw --europepmc-mode multi --no-europepmc-synonym --europepmc-candidate-limit 30 --retrieval-limit 10 --output-dir outputs\document_retrieval --no-resume
python -m evaluation.run_document_retrieval --benchmark benchmark/provideq_benchmark.json --num-questions 5 --seed 42 --retriever fusion --query-strategy raw --paperclip-ranking hybrid --paperclip-candidate-limit 30 --europepmc-mode multi --no-europepmc-synonym --europepmc-candidate-limit 30 --retrieval-limit 10 --rrf-k 60 --output-dir outputs\document_retrieval --no-resume
~~~

Summarize all document-retrieval runs with:

~~~cmd
python -m evaluation.summarize_document_retrieval
~~~

Use Recall@10 as the primary document-retrieval measure, then MRR@10 and
Recall@5. Review errors and fallback warnings before accepting a score.

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
