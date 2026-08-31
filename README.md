# tracelab

Reference implementation and benchmarks for the paper **"Parsing the Stream: A Live
Trace Model for Long-Horizon Agents and Their Observers"** (Egor Pakhomov, Erik
Nijkamp — Salesforce AI Research). arXiv link: TBA-on-announcement.

A long-horizon agent run produces one artifact — its trace — and two consumers of it:
the human observer and the agent itself. tracelab parses the stream once into an
append-only typed ledger, folds it into deterministic run state, and compiles
per-consumer views: an observer page and a compact worker view served back to the
agent by a curator loop.

## Layout

- `src/tracelab/` — ledger, adapters, fold/RunState, derived nodes, views, curator,
  LLM client, benchmark harnesses (DETECT, FIDELITY, COMPREHEND, CONTINUE).
- `tests/` — 99 regression and property tests; every requirement in the paper that is
  test-pinned lives here.
- `tools/recount_oracle.py` — the from-scratch raw-JSONL recount described in
  Appendix B (fold-vs-oracle, 0 mismatches on five chain-120 traces).
- `bench/scoreboard.json` — every experiment variant run during development,
  including failures; run keys match the paper's protocol appendix.
- `bench/spend.json` — the complete per-call spend ledger for every LLM call.
- `bench/continue_v*_traces/` — all recorded CONTINUE workbench run traces (synthetic tasks; variant numbering follows the scoreboard's run keys).
- `bench/synth_corpus/` — the released COMPREHEND corpus: twelve seeded synthetic
  sessions; regenerates byte-identically via
  `uv run python -m tracelab.bench.synth_corpus --seeds 201-212 --out out/`.

Real-corpus modes of FIDELITY/COMPREHEND read the runner's own local Claude Code transcripts (`~/.claude/projects`); nothing is uploaded anywhere by the harness itself beyond the configured model calls. The twelve real transcripts used for the exploratory observer cells are withheld
(personal working sessions); the instrument runs on any reader's own transcripts.

## Setup

```
uv sync --extra dev
uv run pytest -q            # 99 tests
uv run python tools/recount_oracle.py   # fold-vs-oracle recount
```

LLM-backed benchmarks call Anthropic models on Vertex AI; set
`ANTHROPIC_VERTEX_PROJECT_ID` (your GCP project) and `CLOUD_ML_REGION`. Models are
addressed by tier (`claude-sonnet-5`, Haiku 4.5); re-scoring requires the model
endpoints while they remain served.

## Citation

```bibtex
@article{pakhomov2026parsing,
  title={Parsing the Stream: A Live Trace Model for Long-Horizon Agents and Their Observers},
  author={Pakhomov, Egor and Nijkamp, Erik},
  journal={arXiv preprint arXiv:TBA},
  year={2026}
}
```
