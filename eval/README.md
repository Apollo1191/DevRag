# Retrieval Benchmark

This benchmark measures retrieval only. It calls `POST /query` with `use_llm=false`, so OpenAI quota, prompt wording, and answer quality do not affect the retrieval score.

## Run

Start the backend and make sure the indexed repositories include `httpx` and `requests`, then run from the project root:

```powershell
python -m eval.run_retrieval_benchmark
```

Options:

```powershell
python -m eval.run_retrieval_benchmark --top-k 5
python -m eval.run_retrieval_benchmark --base-url http://localhost:8000 --report eval/retrieval_report.json
```

## Metrics

- `hit_at_k`: at least one returned chunk belongs to an expected file or directory.
- `recall_at_k`: fraction of expected path groups represented in the top-k results.
- `mrr`: reciprocal rank of the first relevant result.

The ground truth is intentionally file-level rather than line-level. This keeps the benchmark stable when chunk size or chunk boundaries change.

## Multi-phase RAG benchmark

The larger dataset and runner separate the checks described in the reference architecture:

```powershell
python -m eval.run_rag_benchmark
```

This default run is low-cost: it tests retrieval, out-of-domain behavior, casual-message handling, and latency without calling the LLM.

To test answer generation, opt in explicitly:

```powershell
python -m eval.run_rag_benchmark --with-llm
```

The LLM mode calls the configured provider once per case, so use it only when API credit is available. The report includes retrieval metrics, no-irrelevant-source behavior, answer presence, citation presence, and latency.

## Best practices applied

- Keep retrieval evaluation independent from LLM evaluation.
- Use `top_k` and context budgets to control token spend.
- Keep a small deterministic smoke set for every code change.
- Include casual and out-of-domain questions to measure hallucination risk.
- Store file-level ground truth so benchmark results survive chunk-boundary changes.
- Run LLM evaluation separately and save the report for comparison instead of calling the provider during every development test.

## Adding a repository

Ingest the repository, then append questions to `eval/questions.jsonl`. Each case needs an id, the expected `source_repo` name, a question, and one or more relevant path prefixes:

```json
{"id":"new-case","repo":"repo-name","question":"How does ...?","relevant_paths":["src/module.py"]}
```
