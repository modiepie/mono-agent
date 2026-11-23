# Interview notes

Likely questions about mono-agent with short answers. Numbers come from
[eval/report.md](../eval/report.md); `Dnn` points to [decisions.md](decisions.md).

## The project in 60 seconds

A LangGraph assistant over the monobank Personal API. A router plans the steps; a ReAct agent
answers money questions with seven validated tools over a SQLite cache of the statement; a RAG
step answers tariff questions with hybrid retrieval and citations; a composer writes the
answer, and code checks every number in it against tool results. It runs locally (Ollama,
Qdrant, Langfuse) on synthetic data by default. A 77-question eval with code-checked metrics
compares providers: best setup 91% pass, safety 100% for every model. The most valuable part
was reading the eval's failures: they found real tool bugs that unit tests didn't.

---

## Architecture and LangGraph

**Why LangGraph and not a plain agent loop?**
I needed explicit control flow (guardrail before any model call, a router, a bounded tool
loop, a human-approval pause) plus persistence. LangGraph gives typed state, conditional edges,
checkpointing and `interrupt()` out of the box, and every node shows up as its own span in
traces. A prebuilt `create_react_agent` would hide the loop I wanted to control and explain (D16).

**How does routing work for mixed questions?**
The router returns an intent and an ordered list of steps, e.g. `["transactions", "knowledge"]`.
Each step node removes itself from the plan when done, and one routing function sends the graph
to the next step or to the composer. No combinatorial edges (D16).

**What is in the state, and what is persisted?**
`messages` (questions and final answers) is checkpointed per thread. Everything else (plan, tool
results, ReAct scratchpad, retrieved docs, citations) is per turn and reset by the guardrail.
Tool chatter stays out of the conversation, so later prompts stay small and old tool output
can't be mistaken for fresh data (D17).

**How do you stop an agent from looping forever?**
At most 8 tool rounds per turn, an output cap of 2048 tokens per call and client timeouts. A
looping model without an output cap generates until its context is full, and an inactivity
timeout never fires because each streamed chunk resets it (D16, D39).

**Human-in-the-loop?**
"Export a report" pauses the graph with `interrupt()`; the state is saved in the SQLite
checkpointer, the API sends a `confirm` event, and `/resume` continues with yes or no, even from
another process minutes later. That's the same pattern any real action (payment) would need (D20).

## Tools and function calling

**Why not let the model write SQL?**
Text-to-SQL over financial data is an injection and correctness risk. Tools take Pydantic
models with enums; SQL is fixed fragments plus bound parameters. A test sends
`%' OR 1=1 --` and gets zero rows (D14).

**What happens when the model calls a tool wrongly?**
Validation errors go back as a tool message, and the model fixes its call. Seen live: Qwen sent
a broken `period_b`, got the error, and corrected it on the next step (D15). Empty results are
errors too: when the model queried October 2024 for 2025 data, the tool said "no transactions,
data covers 2025-…, check the year", and it retried with 2025 (D31).

**What did the eval teach you about tool design?**
That silent empty results are the worst failure. `category=jars` + `kind=spending` returned
nothing, and the answer said "you put 0 UAH into jars". SQLite `LIKE` doesn't fold Cyrillic case,
so «оренда» never matched «Оренда». Rent lived in transfer *comments*, which weren't searchable.
All fixed with tests (D41).

## RAG

**Chunking: what did you choose and why?**
I compared fixed 600-character windows with structure-aware chunks (one per section, with a
`Title > Section` header). Structure-aware wins on MRR (0.90 vs 0.85) and recall@1 (81% vs 76%).
Tariff documents are short self-contained sections, so a section answers one question and
cites cleanly (D25).

**Why bge-m3?**
Strong multilingual results including Ukrainian, 8k context, no query/passage prefixes, local
and private. Its cross-lingual ability is visible in the data: English questions retrieve
Ukrainian chunks; BM25 finds nothing for them (D26).

**Did hybrid search help?**
Not on this corpus, and I report that. Dense alone: recall@3 100%, MRR 0.90. Hybrid with RRF
but no reranker: MRR 0.65, because BM25 is noise for cross-lingual questions. The reranker
restores 0.90. I kept hybrid + rerank as the default because BM25's strength (exact codes,
numbers, product names) needs a bigger real corpus to show. The plan is to re-run the ablation
on the real documents and switch to dense if it still wins (D29, D42).

**Why RRF?**
It fuses rankings by rank only: no need to calibrate cosine similarity against BM25 scores;
k = 60 as in the original paper (D29).

**How do you prevent made-up citations?**
Citations are parsed from the answer and kept only if that chunk was actually retrieved (D30).

## Evaluation

**How did you build the dataset?**
77 questions covering totals, comparisons ("why"), ambiguous periods, subscriptions, FX,
refunds, forecasts, knowledge (incl. cross-lingual), mixed, export, out-of-scope and prompt
injection. Expected numbers are computed from the generator by separate code, never typed by
hand, so a bug in the tools can't hide in the expected answer (D32, D33).

**Which metrics, and why mostly not LLM-as-judge?**
Routing, tool name and arguments, exact numbers, recall@k/MRR and safety are all checked by code.
The LLM judge only scores faithfulness and helpfulness, the things code can't check, like
correct numbers with wrong reasoning ("no car costs in October" when there were ₴23,050) (D34).

**What are the weaknesses of your eval?**
It's small: 1–2 questions between configs is noise. The judge is Qwen grading its own answers.
The v2 fixes were found on the same questions, so 73% → 87% is optimistic. The next step is a
held-out set and an independent judge. I wrote these caveats into the report itself (D34, D41).

**How does CI test an LLM app without an LLM?**
Record/replay: model responses and retrieval results for a smoke subset are recorded once and
replayed through LangChain's cache hook. The whole graph runs in CI with no network. A prompt
change shows up as "not recorded", which is a useful signal. Replay deserializes only
message classes, because recordings are files anyone could edit in a PR (D36).

**What did you learn from eval failures?**
That reading the failures matters more than the score. Of 22 baseline failures, 3 were my
dataset being too strict (a better tool call marked wrong), 9 were routing mistakes from gaps
in the router prompt, and most of the rest were the agent struggling with the tools, which
pointed to 4 real tool design bugs (D41).

## Providers, cost, latency

**How do you compare providers?**
Same dataset, same judge, one YAML per setup with a model per node. Results: Gemma routing +
Qwen agent 91%, Qwen alone 87%, Gemma alone 82%. Gemma routes better, Qwen calls tools better, so
combining them wins. Free Nemotron (OpenRouter) matched the local models on the core subset at
about twice the latency (D37, D42).

**How do you track cost?**
A LangChain callback records tokens per model for every call in a request; cost = tokens × a
price table. An unknown price is reported as "n/a", never as $0 (D34).

**What if a provider goes down?**
`FALLBACK_MODEL` wraps every node's model with `with_fallbacks`, bound the same way (tools,
output schema) as the primary. A test makes every primary fail and checks the answer still
comes out (D45).

**Where does the time go?**
Langfuse shows it per node. In Docker on a Mac, retrieval took 5.4 s because containers have
no GPU and the embedding model and reranker ran on CPU; the composer took about 9 s. Fixes in
production: a GPU embedding service or an API embedding model; a smaller composer model (D47).

**Did you fine-tune anything?**
A router. QLoRA on 4-bit Qwen3 0.6B with mlx-lm (bitsandbytes needs CUDA), 3 minutes on a
laptop, trained on template questions written apart from the eval. It went from 61% (prompted)
to 88%, against 90% for the prompted 35B production router, at 71 ms instead of 1.2 s. I didn't
add templates for the questions it missed, because the eval is my only test set. I also didn't
ship it: routing is ~9% of latency, so the next step would be a cascade where the small model
routes only when it's confident (D53).

**What's the MCP server for?**
The same seven tools for any MCP client (Claude Desktop, an IDE, another agent). It's ~40 lines
because schemas, validation and execution come from the tool registry; the safety lives in
the tools, so MCP clients get the masking and sanitizing too (D52).

## Safety and privacy

**Prompt injection?**
Handled in code at two boundaries: the user's message (refused before any model call) and data
(merchant names, transfer comments, documents), where instruction-like sentences are removed
before they reach a prompt. Tested with an injected merchant name, a transfer comment
containing a card number, and a sentence planted in a document. 100% safety in every config.
Regex misses paraphrases; defence in depth still holds: tools are read-only, SQL is
parametrized, and card data is already masked (D18).

**Card numbers and IBANs?**
Masked when data enters the cache, so prompts, traces and answers only ever see `****1234`
(D10).

**How do you keep real data out of the repo?**
Synthetic data by default; the eval forces synthetic mode whatever `.env` says; caches,
checkpoints (incl. SQLite side files) and exports are gitignored; the token is a `SecretStr`
and is never logged (D3, D38).

## monobank API

**How do you handle the rate limits?**
A token bucket per endpoint (1 call / 60 s) that waits only as long as needed; 31-day windows;
backward paging when a page has exactly 500 items; retries with exponential backoff on
429/5xx; and a SQLite cache that tracks fetched ranges, so repeated questions never call the bank
and a range is never marked fetched beyond "now" (D6–D9).

**Why synthetic data, and how realistic is it?**
A public repo can't hold real transactions, and the eval needs a fixed world. The generator
produces a year with real Ukrainian merchants and MCC codes, salary, subscriptions, FX, a
trip in PLN, cashback, refunds, and planted stories (a car-service visit behind an October
taxi spike). It's served through a fake HTTP API with the real limits, so synthetic mode runs
the same client code as real mode (D11).

## Production

**How is the API built?**
FastAPI with Server-Sent Events: progress per graph node, then the answer, which is sent only
after the number check (streaming draft tokens would show text the check might reject). The
graph runs in a worker thread with an event queue. That design came from a real bug: Streamlit
reruns abandon open streams, uvicorn didn't close the generator, the run lock stayed held and
every later request hung (D43, D44).

**What would you change for many users?**
Postgres instead of SQLite for checkpoints and cache, a task queue instead of one run at a time,
an embedding service with a GPU, per-user auth, and per-user data isolation (D44).

**Observability?**
JSON logs (one line per request with latency and tokens) and self-hosted Langfuse v4 traces
with the thread id as session id (D46, D47).

## Your own judgement

**What would you do next?**
A held-out eval set and a stronger, independent judge; a few-shot router prompt (over-routing
to `mixed` is the top remaining failure); a cashback-by-category tool; re-running the RAG
ablation on real documents; support for several cards in one currency; and a cascade router
(the fine-tuned 0.6B model when it's confident, the big model otherwise).

**What was the hardest problem?**
Making "never invent a number" hold even when the model misbehaves. Asking in the prompt is
not enough, so numbers are verified by code, empty results are errors, and the composer only
sees tool outputs. The other hard one was honest evaluation: separating dataset mistakes from
agent mistakes, and reporting results that didn't go my way (hybrid search).

**What mistake did you make?**
I misread a slow eval run as a hang and stopped a healthy run: its progress lines were hidden
because Python buffers piped output. I wrote that into the decision log as it happened, and the
eval runner now flushes progress (D39).
