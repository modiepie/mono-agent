"""Record model responses and retrieval results to JSON, replay them offline (CI).

Uses LangChain's cache hook: in record mode real responses are stored keyed by prompt and model
parameters; in replay mode a missing key is an error (the prompt changed), never a network call.
"""

import hashlib
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core._api import LangChainBetaWarning
from langchain_core.caches import BaseCache
from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from mono_agent.graph.builder import Retrieve


class NotRecorded(RuntimeError):
    pass


class Recording(BaseCache):
    def __init__(self, path: Path, replay: bool) -> None:
        self.path, self.replay = path, replay
        self.data: dict[str, Any] = (
            json.loads(path.read_text("utf-8")) if path.exists() else {"llm": {}, "retrieve": {}}
        )

    @staticmethod
    def _key(*parts: str) -> str:
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:24]

    @staticmethod
    def _stable(prompt: str) -> str:
        """The prompt without per-run noise. LangChain gives every live AI message a fresh run
        id, and Ollama adds timestamps/durations; left in, every prompt after a replayed tool
        call would hash differently from the recorded one."""
        try:
            data = json.loads(prompt)
        except ValueError:
            return prompt
        return json.dumps(
            _drop(data, {"id", "response_metadata", "usage_metadata"}),
            sort_keys=True,
            ensure_ascii=False,
        )

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        stored = self.data["llm"].get(self._key(self._stable(prompt), llm_string))
        if stored is None:
            if self.replay:
                raise NotRecorded("model call not in recording (prompt changed?)")
            return None
        # Recordings are committed files anyone could edit in a PR: revive only the classes a
        # model response consists of, never arbitrary objects (e.g. a model with a new base_url).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            return [
                load(g, allowed_objects=[ChatGeneration, Generation, AIMessage]) for g in stored
            ]

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        if not self.replay:
            key = self._key(self._stable(prompt), llm_string)
            self.data["llm"][key] = [dumpd(g) for g in return_val]

    def clear(self, **kwargs: Any) -> None:
        self.data = {"llm": {}, "retrieve": {}}

    def wrap_retrieve(self, retrieve: Retrieve | None) -> Retrieve:
        def recorded(query: str) -> list[dict]:
            key = self._key(query)
            if key not in self.data["retrieve"]:
                if self.replay or retrieve is None:
                    raise NotRecorded(f"retrieval not in recording: {query!r}")
                self.data["retrieve"][key] = retrieve(query)
            return self.data["retrieve"][key]

        return recorded

    def save(self) -> None:
        if not self.replay:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), "utf-8")


def _drop(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _drop(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_drop(v, keys) for v in obj]
    return obj
