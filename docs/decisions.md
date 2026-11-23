# Design decisions

Each entry: what, why, alternatives considered, trade-offs. Added as decisions are made.

## D1. uv + Python 3.12, `src/` layout

- **What:** uv manages Python, venv and lockfile. Code lives in `src/mono_agent`.
- **Why:** uv is one fast tool for interpreter, deps and lock (`uv.lock` makes CI reproducible). 3.12 is the newest version every ML dependency (torch, sentence-transformers, bitsandbytes) supports well. The `src/` layout forces tests to import the installed package, which catches packaging bugs.
- **Alternatives:** Poetry (slower, no Python management), pip + requirements.txt (no lockfile by default), flat layout (simpler, but tests can accidentally import local files).
- **Trade-offs:** uv is newer than Poetry; fewer people have used it in older teams.

## D2. Dependencies added per milestone

- **What:** `pyproject.toml` starts with only httpx and pydantic; LangGraph, Qdrant client etc. arrive with the code that uses them.
- **Why:** every dependency in the file has a reason I can point to in the code. Smaller install while early milestones are built.
- **Trade-offs:** pyproject changes in several commits instead of one.

## D3. Config via pydantic-settings, token as `SecretStr`

- **What:** one `Settings` class reads env/.env. `MONO_TOKEN` is a `SecretStr`; its absence means synthetic mode.
- **Why:** typed config validated at startup; `SecretStr` hides the token in `repr()`, logs and tracebacks, so it can't leak by accident. One switch (token set or not) decides real vs synthetic, nothing else to configure.
- **Alternatives:** `os.environ` reads scattered across modules (untyped, easy to log by mistake).
- **Trade-offs:** must call `.get_secret_value()` explicitly when building the HTTP header; that is the point.

## D4. SQLite for cache, query layer and checkpointer

- **What:** one SQLite file holds cached statements; tools query it; LangGraph's SQLite checkpointer stores threads.
- **Why:** a single user's statements are thousands of rows, well inside SQLite's comfort zone. Zero ops, works in CI, and the tools can use real SQL aggregation instead of Python loops.
- **Alternatives:** Postgres (needed for multi-user/concurrent writes; overkill here), DuckDB (faster analytics, but a second engine to explain for no measurable gain at this size).
- **Trade-offs:** not suited for many concurrent writers; the migration path to Postgres is the same SQL plus `langgraph-checkpoint-postgres`.

## D5. Synchronous HTTP client (httpx), not async

- **What:** `MonoClient` uses `httpx.Client`.
- **Why:** the API allows one statement call per 60 s, so concurrency buys nothing: requests are serialized by the bank anyway. Sync code is easier to read, test and step through. LangGraph runs sync tools fine, and FastAPI runs sync code in a threadpool.
- **Alternatives:** `httpx.AsyncClient` everywhere.
- **Trade-offs:** if the service ever served many users at once, each blocked sync call would hold a worker thread. The switch to async would be mechanical (same httpx API).

## D6. Token-bucket rate limiter, one bucket per endpoint

- **What:** `TokenBucket(capacity=1, period=60)` for client-info, statement and currency, checked before every request (including retries).
- **Why:** the bank's limit is "1 call per 60 s"; a bucket of capacity 1 enforces exactly that and waits only as long as needed (if the last call was 45 s ago, it waits 15 s). Clock and sleep are injected, so tests run instantly with a fake clock.
- **Alternatives:** a plain `sleep(60)` between calls (wastes time), reacting only to 429 (the bank would see us violate the limit on purpose), a library like `ratelimit` (extra dependency for ~30 lines).
- **Trade-offs:** the lock is held while sleeping, so concurrent callers queue in order. That is intended. The limit lives in process memory: two processes with the same token would not coordinate.

## D7. Retries: 429 / 5xx / network errors, exponential backoff, no jitter

- **What:** up to 5 retries with delays 2, 4, 8, 16, 32 s (capped at 60), or `Retry-After` if larger. 4xx other than 429 fail immediately with `MonoApiError`.
- **Why:** 429 and 5xx are transient; 400/401/403 are our mistakes and retrying can't fix them. Backoff gives the server room.
- **Alternatives:** `tenacity` (fine library, but the loop is short and explicit here), jitter.
- **Trade-offs:** no jitter, because jitter exists to stop many clients retrying in lockstep, and there is one client. Error messages include the path and response body but never headers, so the token can't leak (there is a test for this).

## D8. 31-day windows and 500-item paging inside the client

- **What:** `statement(account, start, end)` splits any range into windows of at most 2,682,000 s. If a response has exactly 500 items, it requests again with `to` = the oldest item's time. Items are de-duplicated by id.
- **Why:** callers (tools, cache) shouldn't need to know API limits. Paging by `to` is what the bank documents. Items that share the boundary second come back twice, hence the de-duplication by id.
- **Edge case:** if more than 500 items share one second, moving `to` doesn't make progress; the client steps back 1 s and those extra items are lost. This is an API limitation, documented in code with a `shortcut:` comment. It cannot happen for a personal account.

## D9. SQLite cache with explicit "covered ranges"

- **What:** fetched items go into `transactions`; each fetched interval is recorded in `synced_ranges`. A sync computes `requested − covered` and fetches only the gaps. Client info (TTL 10 min) and currency rates (TTL 5 min) are cached as JSON in a `kv` table.
- **Why:** "repeated questions never hit the API" needs to know what was fetched, not just what exists: a month with zero transactions is still "fetched". Storing intervals (not just a max timestamp) also handles out-of-order requests (ask about July, then January).
- **Detail:** a range is only marked covered up to *now*, because future transactions can still appear. Rows are upserted (`REPLACE`), so a re-fetched item that went from hold to settled overwrites the old one.
- **Trade-offs / known gap:** items on hold inside an already-covered range are not re-checked. Fix when needed: always re-sync the last few days.

## D10. Mask PANs and IBANs at ingestion

- **What:** `mask_text` replaces 13–19-digit card numbers with `****1234` and Ukrainian IBANs with `UA****1234`. It runs before data enters the cache, so everything downstream (tools, LLM prompts, traces) only ever sees masked values.
- **Why:** masking at the single entry point is much safer than remembering to mask at every exit (prompt, log, trace, API response). There is nothing in the app that needs the full number.
- **Alternatives:** masking only before the LLM call (easy to miss a path, e.g. Langfuse traces of tool outputs), Presidio-style NER (heavy, and the formats here are regular).
- **Trade-offs:** a regex can mask a long non-card number (e.g. a 16-digit order id); that's an acceptable false positive. No Luhn check, on purpose: masking a non-card is harmless, missing a card is not.

## D11. Synthetic data: same API shape, served through a fake HTTP API

- **What:** `generate(seed)` produces `ClientInfo`, `StatementItem`s and `CurrencyRate`s for calendar year 2025: 3 accounts (UAH black card with credit limit, USD, EUR), 2 jars, about 900 items. `FakeMonoApi` serves them through `httpx.MockTransport` and enforces the real limits (31-day window, newest 500, token required).
- **Why:** synthetic mode exercises the exact same client, paging and cache code as real mode. The only branch on mode is `mono/factory.py`. A fixed year (not "last 12 months") makes eval answers stable forever. `random.Random(seed)` with no set/hash iteration makes output identical across runs and machines.
- **Realism:** real Ukrainian merchant brands and MCC codes; fictional people and employer. Salary with a mid-year raise, rent, utilities with winter heating, subscriptions in UAH and USD, FX purchases, a trip paid in PLN from the EUR account, cashback on chosen categories with monthly payout, P2P transfers, last two days on hold.
- **Planted stories for the eval:** October taxi spike explained by a car-service visit (no fuel purchases Oct 3–17) plus a 15% tariff rise; overlapping music subscriptions; MEGOGO cancelled; Netflix price increase; duplicate charge + refund; a return; an outlier laptop purchase; unknown foreign PayPal charges; prompt-injection text in a merchant name and in a transfer comment (with a card number and IBAN that must be masked).
- **Trade-offs:** hand-written behaviour, not learned from real data, so distributions are plausible rather than statistically faithful. Fine for testing an agent, not for training a fraud model.

## D12. Ruff rules RUF001–003 off

- **What:** ruff's "ambiguous Unicode character" rules are disabled.
- **Why:** they flag Cyrillic letters that look like Latin ones (`С`, `О`, `а`). In a Ukrainian project those are intentional, and the rule would fire on every merchant name.

## D13. What "spending" means, and categories as a SQLite function

- **What:** MCC codes map to ~25 categories in `tools/categories.py`. Transfers (MCC 4829) are split by description into `transfers`, `jars`, `currency_exchange`, `cashback`. *Spending* = `-SUM(amount)` over everything except those four, so refunds automatically reduce spending. The mapping is registered with `sqlite3.create_function`, so SQL can `WHERE` and `GROUP BY` on it.
- **Why:** "how much did I spend" must not count rent paid to a person, money moved to a jar or buying dollars as consumption; and a refunded purchase shouldn't count twice. One Python function is the single definition used by SQL, tools and tests.
- **Alternatives:** a `category` column filled at ingestion (needs a migration when the mapping changes), a SQL `CASE` (duplicates the mapping), an LLM categorizer (non-deterministic; it's the QLoRA stretch goal instead).
- **Trade-offs:** rent paid by card transfer counts as a transfer, not as housing spending. That's defensible, and I would say so in the answer.

## D14. Parametrized query builder, never LLM-written SQL

- **What:** tools accept Pydantic models with enums (`Literal`) for categories, group-by fields and sort orders; SQL is assembled from fixed fragments with `?` parameters. `LIKE` wildcards in merchant search are escaped. The three aggregation tools from the spec are one `aggregate_spending(group_by=...)`.
- **Why:** text-to-SQL is an injection and correctness risk on financial data. An enum the model can't escape is a hard guarantee; a test sends `%' OR 1=1 --` and gets zero rows. One aggregate tool with `group_by` gives the model fewer, clearer choices than three near-identical tools.
- **Alternatives:** LangChain SQL agent (flexible, but unbounded queries and hard to evaluate), separate tools per grouping.
- **Trade-offs:** questions outside the tool set ("median purchase on weekends") can't be answered. That's acceptable: the eval tells which tools are missing.

## D15. Tool registry; validation errors go back to the model

- **What:** `TOOLS` maps a name to (description, input model, function). `run_tool` validates with Pydantic and returns `(output, ok)`. An invalid call becomes an error `ToolMessage`, not an exception.
- **Why:** small models produce a malformed call now and then. Seen live: Qwen sent `period_b` as a broken string, received the validation error, and corrected itself on the next step. Tool schemas are generated from the same Pydantic models (refs inlined), so the schema the model sees and the validator can't drift apart.
- **Alternatives:** `@tool`-decorated functions (more magic, less explicit validation and error control).

## D16. Graph: router plan + explicit ReAct loop

- **What:** `guardrail → router → steps → composer`. The router (structured output) returns an intent and an ordered list of steps (`transactions`, `knowledge`, `export`). Each step node removes itself from the plan when done; a routing function sends the graph to the next step. The transactions agent is a hand-written ReAct loop: `agent ⇄ tools`, capped at 8 rounds.
- **Why:** a plan list supports mixed questions ("my cashback + the cashback rules") without a combinatorial number of edges. The explicit loop is ~30 lines, every step is visible in the graph and in traces, and the step cap is mine to control.
- **Alternatives:** `create_react_agent` / `create_agent` prebuilt (less code, but hides the loop I need to explain and record), a supervisor LLM that decides after each step (more flexible, more LLM calls, more ways to fail).
- **Trade-offs:** the plan is fixed after routing; the graph can't decide halfway that it also needs documents. Fine for these question types.

## D17. State: conversation vs. per-turn data

- **What:** `messages` (questions and final answers) is checkpointed across turns. Everything else (plan, tool results, scratchpad of tool calls, docs, citations) is per turn and reset by the guardrail node at the start of each turn. ReAct messages live in `scratchpad`, not in `messages`. The graph has a separate input schema: callers send only messages.
- **Why:** keeping tool chatter out of the conversation keeps later prompts small (cost, latency) and stops old tool output from being mistaken for fresh data. Overwrite semantics for per-turn fields avoid reducer tricks for resetting lists.
- **Trade-offs:** a follow-up ("and in November?") re-runs tools instead of reusing previous results. The SQLite cache makes that cheap.

## D18. Prompt-injection defence in code, at two boundaries

- **What:** (1) the user's message is checked with a regex for instruction-override patterns (EN/UK/RU) before any model call, and refused. (2) Every merchant name and comment leaving a tool passes through `sanitize`, which replaces instruction-like text with `[removed: instruction-like text]` and sets `suspicious_text_removed`. Prompts also say tool output is data.
- **Why:** prompts alone are not a defence. The model reads whatever we put in front of it. Data-side sanitization means the planted "IGNORE ALL PREVIOUS INSTRUCTIONS" merchant never reaches a prompt at all (there's a test that checks the composer prompt). Regex is deterministic, free and explainable.
- **Alternatives:** an LLM or classifier guard (e.g. Llama Guard, a fine-tuned detector): catches paraphrases, but costs a call and can itself be fooled.
- **Trade-offs:** a paraphrased injection gets past the regex. Defence in depth still holds: tools can't write anything, SQL is parametrized and card numbers are already masked. The eval set includes injection cases to measure this.

## D19. Numbers in the answer are checked against tool results

- **What:** the composer's answer is parsed for numbers (handles `5 863,87`, `1,360.96`, `-250`, `30%`). Each must match a number in the tool results, their arguments or the question (allowing rounding and sign). If not, the composer gets one rewrite request listing the bad numbers; if it still fails, the answer carries a visible warning.
- **Why:** "refuses to invent figures" should be enforced, not just requested. This is cheap and catches the most damaging failure in a finance assistant.
- **Limits:** small integers (≤ 31) pass unchecked: counts, days, "3 months" (a `shortcut:` comment marks this). It checks numbers, not reasoning. Seen live: all numbers grounded, yet the answer said "no car expenses in October" when there were ₴23,050. That is why M4 adds an LLM-judge faithfulness metric.

## D20. Human-in-the-loop export with `interrupt()` and a SQLite checkpointer

- **What:** "export a report" routes to `transactions → export`. The export node calls `interrupt()`, the graph state is saved to the checkpointer, and the caller (CLI now, API later) asks the user; `Command(resume=True)` writes a Markdown report, `False` writes nothing.
- **Why:** nothing in this app writes to the bank, but writing a file with financial data is an action the user should confirm, and it's the pattern any real action (payment, transfer) would need. With a checkpointer the pause survives process restarts: approval can come minutes later from a different request.
- **Alternatives:** asking in the composer text and parsing the next message (fragile), no confirmation.

## D21. One model string per node; local Qwen by default

- **What:** `ROUTER_MODEL`, `AGENT_MODEL`, `COMPOSER_MODEL` are `provider:model` strings passed to LangChain's `init_chat_model`. The default is `ollama:qwen3.6:35b-a3b-q4_K_M` for all three, with Qwen's thinking mode off and temperature 0.
- **Why:** per-node choice is what makes "small model for routing, larger for answers" a config change, not a code change, and it's exactly what the provider comparison in the eval needs. Qwen3.6 35B-A3B is a mixture-of-experts model (~3B active parameters), so it runs fast on a laptop and handles tool calling well; it's also multilingual (Ukrainian). Thinking off: about 2× faster, and tool calls didn't need it in testing.
- **Trade-offs:** `init_chat_model` needs each provider's integration package installed. Only `langchain-ollama` is added now; the others come with the provider comparison (M4).
- **Today's date:** the agent's "today" is fixed at 2025-12-31 in synthetic mode, so "October" always means October 2025 and eval answers never drift. Real mode uses the real date.

## D22. Graph tests with a scripted fake model

- **What:** `tests/conftest.py::Scripted` is a `BaseChatModel` that returns pre-written messages in order and records every prompt. Router decisions are scripted as tool calls, which LangChain's default `with_structured_output` parses.
- **Why:** graph logic (routing, loop bounds, retries, refusals, interrupts, memory) must be tested deterministically and offline. Recording prompts lets tests assert what the model *saw*, e.g. that injected text never reached the composer.
- **Trade-offs:** these tests prove the wiring, not answer quality. Quality is the eval's job (M4).

## D23. Placeholder documents, clearly marked

- **What:** `data/docs` ships six small HTML documents (cashback, credit limit and grace period, black-card tariffs, currency accounts, jars, an English FAQ). Each says "Зразок для демонстрації. Не є офіційними умовами monobank" and has `<meta name="sample" content="true">`, which travels into every chunk's metadata.
- **Why:** the RAG pipeline and its eval need a fixed corpus now; the real PDFs/HTML get added later. Tariff numbers in these files are illustrative, and the marking makes sure no one mistakes them for real terms. The English FAQ plants an injection sentence to test document-side sanitization.
- **Trade-offs:** knowledge-question eval answers are tied to this corpus; when real documents replace it, the knowledge part of the eval dataset must be updated too.

## D24. Loaders keep the heading path; injections are dropped at load time

- **What:** HTML is parsed with BeautifulSoup: a heading stack (`h1`–`h4`) gives every paragraph, list item and table row its section path; table rows become `cell | cell` lines. PDFs use pypdf; numbered lines ("2.1 …") and short ALL-CAPS lines are treated as headings. Sentences that look like instructions to an AI are removed when the document is loaded, keeping the rest of the paragraph.
- **Why:** the section path is what structure-aware chunking and citations need. Removing injected sentences at ingestion follows the same rule as card masking (D10): clean once at the entrance, so nothing downstream has to remember.
- **Limits:** PDF heading detection is heuristic; scanned PDFs (images) would need OCR, which isn't included.

## D25. Two chunking strategies, compared rather than assumed

- **What:** *fixed*: a 600-character sliding window with 100 overlap over the whole document, cut at word boundaries (15 chunks for this corpus). *structure*: one chunk per section, long sections split at paragraph then sentence boundaries up to 1000 chars, and the embedded text prefixed with `Title > Section` (38 chunks).
- **Why:** fixed-size is the standard baseline. Tariff documents are made of short, self-contained sections ("Мінімальний платіж", "Коли пільговий період не діє"); a chunk that matches a section answers one question and cites cleanly. The header prefix lets a short chunk like "5% від суми заборгованості…" still match "minimum payment".
- **Trade-offs:** structure-aware chunks can be tiny (the title/banner chunk), and the comparison changes two things at once (boundaries and header). That is what "structure-aware" means in practice. The M4 ablation measures recall@k and MRR for both.

## D26. Embeddings: bge-m3 locally, an API model as the alternative

- **What:** `EMBEDDINGS=local:BAAI/bge-m3` (sentence-transformers, on Apple MPS here). Any other value (`google_genai:gemini-embedding-001`, `openai:text-embedding-3-small`) goes through LangChain's `init_embeddings` and needs that provider's package and key. Models load from the local Hugging Face cache by path, so startup makes no network calls.
- **Why bge-m3:** trained on 100+ languages with strong Ukrainian results, up to 8k-token inputs, no query/passage prefixes to get wrong, free and private (documents never leave the machine). Observed: an English question ("How does the credit limit grace period work?") retrieves the right Ukrainian chunk; BM25 returns nothing for it.
- **Alternatives:** multilingual-e5-large (good, but needs `query:`/`passage:` prefixes and has a 512-token limit), OpenAI or Gemini embeddings (no local compute, but cost per call and data leaves the machine), Ollama-served bge-m3 (avoids PyTorch, but the reranker needs PyTorch anyway).
- **Trade-offs:** about 2.3 GB of weights and PyTorch in the install. The API option is wired but not installed until there's a key to compare with.

## D27. Qdrant, one collection per (chunking, embedding model)

- **What:** Qdrant v1.20 in Docker Compose (client and server versions match). Collections are named `mono_docs_<strategy>_<embedding>`; `mono-agent index` rebuilds both strategies from `data/docs`. Tests use Qdrant's in-memory mode (`QdrantClient(":memory:")`), so CI needs no Docker.
- **Why:** a real vector database with payloads (chunk text and metadata come back with every hit), filtering, and a server process, like production. Separate collections per configuration let the ablation switch with one setting, and vectors from different models never mix.
- **Alternatives:** Chroma or FAISS (lighter, but weaker as a "production" story), pgvector (great if Postgres is already there; it isn't).
- **Trade-offs:** rebuilding the whole collection on every index is fine for tens of chunks; for thousands, upsert by content hash instead.

## D28. BM25 written in ~40 lines, with prefix stemming for Ukrainian

- **What:** Okapi BM25 (k1 = 1.5, b = 0.75) in Python over the same chunks Qdrant holds (scrolled at startup: one source of truth). Tokens are lowercased words with stopwords removed and truncated to 6 characters, a crude stemmer.
- **Why:** Ukrainian is highly inflected (кешбек, кешбеку, кешбеком) and has no Snowball stemmer, so the standard BM25 tokenizers (Qdrant/fastembed use Snowball) miss most matches. Prefix truncation is a known cheap approximation, and it's easy to explain. At this corpus size, scoring every chunk takes microseconds.
- **Alternatives:** Qdrant sparse vectors with server-side fusion (the way to go at scale), `rank_bm25` (a dependency for the same math), a morphological analyser like pymorphy3 (better stems, another dependency).
- **Trade-offs:** in-memory BM25 doesn't scale to millions of chunks, and prefix truncation sometimes merges unrelated words. Both are marked as shortcuts in code.

## D29. Hybrid = RRF of dense + BM25, then a cross-encoder reranker; each stage switchable

- **What:** `search(query, mode="dense"|"bm25"|"hybrid", rerank=True|False)`. Hybrid takes the top 20 of each and fuses them with reciprocal rank fusion (`Σ 1/(60 + rank)`); the reranker (`BAAI/bge-reranker-v2-m3`) rescores those candidates and the top 5 go to the composer.
- **Why:** dense handles meaning and language switches; BM25 handles exact terms (numbers, "MCC", product names). RRF uses only ranks, so there's no need to calibrate cosine against BM25 scores, and k = 60 is the standard value from the original paper. A cross-encoder reads query and chunk together, so it's much more precise than either retriever, but too slow to run on the whole corpus; 20 candidates keeps it cheap. Observed: for "is cashback paid on EUR account purchases?" only the reranked pipeline puts `currency-accounts#4` first.
- **Trade-offs:** the reranker adds a second 2.3 GB model and about 100–300 ms per query. The ablation shows whether it pays off.

## D30. Knowledge node: standalone-query rewrite only for follow-ups; citations are verified

- **What:** the first message is searched as is; a follow-up ("а для готівки?") is first rewritten into a standalone query by the router model. The composer must cite chunk ids like `[credit-limit#2]`; citations are parsed from the answer and kept only if that chunk was actually retrieved. If Qdrant is down or the index missing, the knowledge step returns no documents and the answer says so; transaction questions still work.
- **Why:** follow-ups are meaningless as search queries on their own, but rewriting every message would cost a model call for nothing. Verifying citations stops made-up references (a test feeds `[made-up#7]` and checks it's dropped).

## D31. Tools return errors the model can act on

- **What:** a period with no transactions at all returns an error naming the cached date range and today's date. `aggregate_spending` rejects non-spending categories with a hint to use `query_transactions`, and reports cashback earned per group.
- **Why:** seen live: asked about "October", the model queried October 2024 (the data is 2025), got zeros, and the composer said "you received 0 грн cashback". A silent zero is worse than an error: an error message is feedback the model can use, and it did (it retried with 2025). Prompt instructions alone didn't prevent it.

## D32. Ground truth computed from the generator, not written by hand

- **What:** `eval/truth.py` computes expected numbers (spend per category and period, counts, cashback, forecasts, FX, rates) with plain Python over the generator's `StatementItem`s. The dataset refers to them (`{fn: spent, period: 2025-10, category: taxi}`), so nobody types "5863.87" into a YAML file.
- **Why:** hand-written expected values drift when the generator changes and hide arithmetic mistakes. Computing them independently from the tools' SQL means a bug in a tool can't also be baked into the expected answer. (The tests do the same.) The only shared piece is the MCC → category mapping, which is the *definition* of a category, not a computation.
- **Trade-offs:** a few expectations are literals where the fact is a planted constant (rent 180 000, car service 23 050); they are cross-checked against the truth functions.

## D33. Dataset: 77 questions, each with machine-checkable expectations

- **What:** `eval/dataset/questions.yaml` covers totals, counts, comparisons and "why", ambiguous periods ("last month", "autumn", "Q3", "October" with no year), subscriptions, FX, refunds, income, cashback, forecasts, anomalies, knowledge (incl. cross-lingual), mixed, export (HITL), out-of-scope, and prompt injection (in user input, in a merchant name, in a transfer comment with a card number and IBAN, and in a document). Each question lists its expected intent, expected tool calls (acceptable names + an argument subset), expected numbers, expected evidence (document + text) and forbidden strings.
- **Why:** each expectation becomes a metric that code can check, so most of the eval needs no LLM opinion at all. Argument *subsets* and lists of acceptable tools keep the eval from punishing correct but different solutions.
- **Lesson from the first run:** the agent searched "the 250 UAH charge on February 20" with `date_from = date_to = 2025-02-20`, which is better than the month I'd expected, and my expectation marked it wrong. Expected arguments must describe what any correct call has to contain, not one particular solution.

## D34. Metrics: code first, LLM judge only for what code can't check

- **Routing accuracy:** intent matches (blocked-by-guardrail counts as "injection").
- **Tool accuracy:** *name* (an acceptable tool ran successfully) and *name + args* (with the expected argument subset); failed calls don't count.
- **Numeric correctness:** every expected number appears in the answer, exact or rounded to whole units, parsed with the same number parser as the grounding guardrail. Never judged by an LLM.
- **Retrieval:** recall@5 and MRR, where a chunk is relevant if it comes from the expected document and contains the expected text. That makes the metric independent of chunk ids, so the two chunking strategies are comparable.
- **Safety:** refusals where required; forbidden strings (injected claims, PAN, IBAN) absent.
- **LLM judge** (`eval/judge.py`): faithfulness and helpfulness 1–5 against a rubric, given the same data the composer saw plus the author's reference note. It catches wrong *reasoning* with right numbers (D19). The same judge model is used for every config, so scores are comparable.
- **Pass:** a question passes only if every applicable code check is perfect. Strict on purpose: "4 of 5 numbers right" is a wrong answer in finance.
- **Cost and latency:** a LangChain callback (`providers/usage.py`) records tokens per model for every call in a request; cost = tokens × the config's price table. Unknown prices are reported as "n/a", never as $0. The same tracker is meant for the API service (M5).
- **Known biases:** the judge is a local Qwen judging (among others) Qwen's own answers, a self-preference risk; a stronger, different judge (e.g. Claude Sonnet) is the fix once an API key is available. The dataset is small, so a difference of 1–2 questions between configs is noise.

## D35. RAG ablation runs retrieval only

- **What:** `--ablation` runs every evidence question through {fixed, structure} × {BM25, dense, hybrid, dense+rerank, hybrid+rerank} and reports recall@1/3/5, MRR and latency. No LLM involved.
- **Why:** retrieval quality can be measured cheaply and deterministically. Mixing it with generation would blur which stage helped.

## D36. Record/replay for CI

- **What:** `--record` stores every model response and retrieval result for the smoke subset in `eval/recordings/*.json` through LangChain's cache hook; `--replay` serves them back and fails on any call that isn't recorded. `tests/test_eval_smoke.py` replays the smoke subset and requires the same verdicts and the same answers as the committed baseline.
- **Why:** CI can't run a 23 GB model, and paid API calls on every push are slow, flaky and cost money. Replaying the full graph still exercises routing, tools, guardrails, composer checks and metrics, with zero network. A changed prompt shows up as "not recorded", which is a useful signal on its own: re-record and review the new answers.
- **Security:** recordings are committed files anyone could edit in a PR, so replay deserializes only `ChatGeneration`/`AIMessage` objects (LangChain's `load` with an explicit allowlist), never arbitrary classes.
- **Trade-offs:** replay tests the code around the model, not the model. Quality regressions are caught by re-running the full eval.

## D37. Provider configs: one YAML per setup, same dataset, same judge

- **What:** `eval/configs/*.yaml` names a model per node (router, agent, composer) plus the judge and a price table. Local: `local-qwen` (Qwen3.6 35B-A3B everywhere), `local-gemma` (Gemma 4 everywhere), `local-mixed` (Gemma routes, Qwen does the rest). API: `anthropic` (Haiku 5.5 routes, Sonnet 5.5 answers; list prices as of 2026-11), `gemini`, `openrouter`. A config whose API key is missing refuses to start with a clear message.
- **Why:** "small model for routing, larger for answers" is then a measured claim, not a slogan. Claude models get no `temperature` (current models reject non-default sampling); Ollama gets `reasoning=False`.
- **Gaps:** the Gemini and OpenRouter model ids and prices must be verified before use; I didn't guess prices.

## D38. The eval never touches real data

- **What:** the runner builds its own `Settings` with `mono_token=None` and an in-memory cache, whatever `.env` says.
- **Why:** results, recordings and the report are committed. A misconfigured machine with a real token must not leak real transactions into the repository.

## D39. Every model call has an output cap and a bounded context

- **What:** Ollama models run with `num_predict=2048` and `num_ctx=32768`; Anthropic with `max_tokens=2048`; every client has a timeout and retries.
- **Why:** without an output cap, a model that falls into a repetition loop keeps generating until its context is full, and a client timeout doesn't stop it: LangChain streams from Ollama internally, and an inactivity timeout resets with every chunk. So a timeout alone doesn't bound a call; an output cap does. Ollama also loaded Qwen with its full 262k-token context by default (28 GB); our largest prompt is about 12k tokens, and at 32k the model takes 23 GB.
- **Honest note:** I added this after misreading a slow eval run as a hung one. The run was healthy; its progress lines just weren't visible because Python buffers stdout when piped. The cap is still the right default, and the eval runner now flushes every progress line.
- **Trade-offs:** an answer longer than 2048 tokens gets cut. That's far above anything this assistant should say. In M5 an overall per-request deadline at the API layer becomes the second safety net.

## D40. Rate-limited questions are "not run", not "wrong"; runs can resume

- **What:** an error caused by a rate limit or the network marks the question as *not run*. It's excluded from the scores and shown as a separate count. `--resume` keeps finished answers and re-runs only those.
- **Why:** OpenRouter's free tier allows about 50 requests per day. The free Nemotron run hit it at question 12 of 20. Counting those 8 as failures would rank a model on infrastructure, not quality; dropping them silently would hide that the run is incomplete.
- **Core subset:** free tiers run only `--subset core` (every 4th question, a stratified sample of 20). The report then compares every config on the same 20 questions.

## D41. Fixes driven by the first full eval run (baseline → v2)

The baseline `local-qwen` run passed 55 of 77 questions (71%). Reading every failure, not just the score, sorted them into three kinds:

- **The dataset was wrong (3).** Correct answers reached by valid but different routes were marked as failures: comparing all categories instead of one, querying Nov 20–30 instead of all of November. "Скільки кешбеку я отримав у жовтні" is genuinely ambiguous (earned on October purchases vs paid out in October). Fixes: looser argument subsets, and an `any:` list of acceptable values for ambiguous numbers. `--rescore` re-scores stored answers against the corrected dataset without new model calls: the baseline moved from 71% to 73% on fairness alone.
- **Tool design bugs (found by real failures).** 1) `category=jars` + `kind=spending` silently returned nothing, and the answer said "you put 0 UAH into jars". Contradictory filters are now a validation error with a hint (`kind=all`). 2) Rent is a transfer whose *comment* says «Оренда квартири»; search covered only the description. The filter is now `text` and searches both. 3) SQLite `LIKE` folds case only for ASCII, so «оренда» never matched «Оренда»: replaced by a Python `casefold` function registered in SQLite. 4) "Which purchases were refunded" had no filter: `kind=refunds` added.
- **Prompt gaps.** The router sent currency rates and subscription prices to `out_of_scope`/`knowledge`, and savings forecasts to `mixed`. Its prompt now lists explicitly what counts as the user's own data (rates, subscriptions, forecasts, "why" questions). The agent prompt now says: always call a tool, money moves (`transfers`/`jars`/`currency_exchange`/`cashback`) are queried with `kind=all`, cashback earned lives in `aggregate_spending`, and bank texts are Ukrainian.
- **Overfitting caveat:** the fixes were found on the same 77 questions they are measured on, so v2's gain is optimistic. I kept the fixes general (tool semantics, a category list in the router) rather than question-specific wording. The proper next step is a held-out set written after these fixes.
- **Checkpointing:** the runner saves results after every question, so any interruption (rate limit, crash, a laptop going to sleep) costs one question, not the whole run. Learned the hard way: a 77-question local run on a laptop on battery with the lid closed advanced only during macOS maintenance wake-ups, about one model call every 17–30 minutes. `caffeinate -i` prevents idle sleep, not closed-lid sleep on battery. Long local evals need AC power and an open lid. The latency numbers survive a sleep: `time.perf_counter()` on macOS doesn't advance while the machine sleeps, so the measured times count only time awake (checked: the slowest question in that run took 125 s of real compute). `--resume --only <ids>` re-runs single questions when needed.

## D42. What the eval says (v2, synthetic data, placeholder docs)

Full numbers: `eval/report.md`. Read with the caveats in D34 (small dataset; Qwen judges itself) and D41 (fixes measured on the same questions).

- **Best setup: Gemma 4 routes, Qwen3.6 does the rest: 91% pass** (Qwen alone 87%, Gemma alone 82%). Gemma routes better (94% vs 91%) but makes more tool and number mistakes (81% numbers vs 93%). Qwen is the better tool caller but sends too many questions to `mixed`. Per-node model choice (D21) pays off: this is the "small model for routing, larger for answers" idea, except the measured winner isn't the smaller one, it's the one that routes better.
- **Code-checked safety holds: 100% on all 12 safety questions in every config** (out of scope, injection in input, merchant names, comments and documents, PAN/IBAN never leaked). That comes from code (D18, D10), not from model behaviour, which is why it doesn't vary by model.
- **Where it still fails:** over-routing to `mixed` (an extra retrieval step: slower and more tokens; the eval counts it as a routing error even when the answer is right), questions that need summing across several transactions (salary raise, ChatGPT in USD), and mixed questions (40–60%). Next fixes would be a few-shot router prompt and a tool that sums cashback by category for a period.
- **Free API model:** Nemotron 3 Ultra (OpenRouter free tier) passed 10 of the 12 core questions it could run before the daily cap: on par with the local models, at about twice the latency (27 s vs 13 s per question). Free tiers fit quick comparisons, not a full eval.
- **Cost:** $0 for all configs run (local or free tier). The cost column and price tables are wired (D34), so a paid config (Anthropic) fills it with real numbers once a key is set.
- **RAG ablation:** with structure-aware chunks, dense retrieval alone already reaches 100% recall@3 and MRR 0.90. Hybrid without the reranker is *worse* (MRR 0.65): BM25 finds nothing for English questions about Ukrainian documents and drags good dense results down in the fusion. The reranker fixes that, but adds no gain over dense alone on this corpus. Structure-aware beats fixed-size on MRR (0.90 vs 0.85) and recall@1 (81% vs 76%). Decision: keep the default `structure` + `hybrid` + `rerank`. It equals dense here, and BM25's strength (exact terms like product codes and numbers) only shows on a bigger, real corpus. Re-run the ablation when the real tariff documents are added; if dense still wins, switch the default to dense and skip ~250 ms of reranking per question. (The 237 ms for the first fixed/dense row includes model warm-up.)
