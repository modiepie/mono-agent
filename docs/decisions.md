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
