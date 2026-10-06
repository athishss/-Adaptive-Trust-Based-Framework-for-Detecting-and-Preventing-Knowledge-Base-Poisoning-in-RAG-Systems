"""LLM backends.

``StubLLM``      - deterministic, offline, used by tests and the demo.
``OpenAICompatLLM`` - any OpenAI-compatible server: vLLM, llama.cpp, the API.
``OllamaLLM``    - local Ollama daemon (laptop demo).
``HFLocalLLM``   - transformers in-process (needs torch).

All of them run greedy (temperature 0) with a fixed seed where the backend
supports it, because every reported number has to be reproducible.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

from ..contracts import LLMResponse
from ..utils.textnorm import normalise, sentences, tokenise
from ..utils.timing import Stopwatch


class LLMError(RuntimeError):
    pass


class StubLLM:
    """Offline stand-in that actually reads the context.

    It picks the context passage with the highest lexical overlap with the
    question and returns its most relevant sentence with a correct citation.
    For common ``who`` questions it also requires the sentence to contain the
    queried relation (for example, ``designed``); sharing only the subject is
    not enough to claim an answer. That keeps topic-adjacent passages from
    masquerading as evidence in offline verification tests.
    """

    name = "stub"
    _WHO_RELATIONS = frozenset({
        "built", "build", "designed", "design", "discovered", "discover",
        "found", "painted", "paint", "formulated", "formulate", "invented",
        "invent", "created", "create", "founded", "found", "authored",
        "wrote", "directed", "composed", "developed", "proposed", "walk",
        "walked", "landed", "led", "made", "constructed", "construct",
        "erected", "named", "identified", "first",
    })

    def __init__(self, abstain_marker: str = "INSUFFICIENT EVIDENCE") -> None:
        self.abstain_marker = abstain_marker
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: Optional[Sequence[str]] = None) -> LLMResponse:
        self.calls += 1
        with Stopwatch() as watch:
            question, blocks = self._parse_prompt(prompt)
            q_tokens = set(tokenise(question))
            best_id, best_sentence, best_score = None, "", 0.0
            for doc_id, text in blocks:
                for sentence in sentences(text) or [text]:
                    overlap = len(q_tokens & set(tokenise(sentence)))
                    score = overlap / max(1, len(q_tokens))
                    if score > best_score:
                        best_id, best_sentence, best_score = doc_id, sentence, score
            if (best_id is None or best_score == 0.0
                    or not self._answers_who_question(question, best_sentence)):
                text = self.abstain_marker
            else:
                answer = self._concise_who_answer(question, best_sentence)
                text = f"{answer.rstrip('.')} [{best_id}]"
        return LLMResponse(text=text, llm_calls=1, latency_ms=watch.elapsed_ms, model=self.name)

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[str, List[tuple[str, str]]]:
        question = ""
        blocks: List[tuple[str, str]] = []
        current_id: Optional[str] = None
        buffer: List[str] = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Question:"):
                question = stripped[len("Question:"):].strip()
                continue
            if stripped.startswith("[") and "]" in stripped:
                if current_id is not None:
                    blocks.append((current_id, normalise(" ".join(buffer))))
                current_id = stripped[1:stripped.index("]")]
                buffer = [stripped[stripped.index("]") + 1:]]
                continue
            if current_id is not None and stripped:
                buffer.append(stripped)
        if current_id is not None:
            blocks.append((current_id, normalise(" ".join(buffer))))
        return question, blocks

    @classmethod
    def _answers_who_question(cls, question: str, sentence: str) -> bool:
        """Reject a topic match that omits an obvious queried relation/value."""
        normalized_question = normalise(question).lower()
        sentence_tokens = set(tokenise(sentence))
        if normalized_question.startswith("who "):
            relation_tokens = set(tokenise(normalized_question)) & cls._WHO_RELATIONS
            return not relation_tokens or bool(relation_tokens & sentence_tokens)
        if re.match(r"^(how high|what is the height|how tall)\b", normalized_question):
            return bool(re.search(r"\d", sentence))
        return True

    @classmethod
    def _concise_who_answer(cls, question: str, sentence: str) -> str:
        """Extract a short answer for common who/how-high smoke-test prompts."""
        normalized_question = normalise(question).lower()
        if re.match(r"^(how high|what is the height|how tall)\b", normalized_question):
            measured = re.search(
                r"\b\d[\d,]*(?:\.\d+)?\s+(?:metres?|meters?|feet|foot|"
                r"kilometres?|kilometers?|miles?)\b",
                sentence,
                flags=re.IGNORECASE,
            )
            if measured is None:
                measured = re.search(r"\b\d[\d,]*(?:\.\d+)?\b", sentence)
            return measured.group(0) if measured else sentence
        if not normalized_question.startswith("who "):
            return sentence
        relations = set(tokenise(normalized_question)) & cls._WHO_RELATIONS
        if not relations:
            return sentence
        alternatives = "|".join(re.escape(word) for word in sorted(relations, key=len, reverse=True))
        passive = re.search(
            rf"\b(?:{alternatives})\s+by\s+((?:[A-Z][A-Za-z'-]*|the|of|and)"
            rf"(?:\s+(?:[A-Z][A-Za-z'-]*|the|of|and))*)",
            sentence,
        )
        if passive:
            return passive.group(1).strip()
        question_terms = set(tokenise(normalized_question))
        name_pattern = re.compile(
            r"\b([A-Z][A-Za-z'-]*(?:\s+(?:[A-Z][A-Za-z'-]*|the|of|and)){0,3})\b"
        )
        for candidate in name_pattern.finditer(sentence):
            name = candidate.group(1).strip()
            name_terms = set(tokenise(name))
            # Ignore title/topic words at the beginning of a passage (the
            # ingestion adapter may prepend a lower-case title before the
            # actual sentence). Keep at least one name token not in the query.
            if name_terms - question_terms:
                return name
        return sentence


class _HTTPBackend:
    """Minimal JSON POST helper (stdlib only, so no requests dependency)."""

    @staticmethod
    def post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:                     # pragma: no cover - network
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:                      # pragma: no cover - network
            raise LLMError(f"cannot reach {url}: {exc.reason}") from exc


class OpenAICompatLLM:
    """vLLM (``vllm serve meta-llama/Llama-3.1-8B-Instruct``) or the OpenAI API."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1",
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0,
                 seed: int = 20260921, timeout_s: float = 120.0) -> None:
        self.name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.timeout_s = float(timeout_s)

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: Optional[Sequence[str]] = None) -> LLMResponse:
        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
            "seed": self.seed,
        }
        if stop:
            payload["stop"] = list(stop)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        with Stopwatch() as watch:
            body = _HTTPBackend.post(f"{self.base_url}/chat/completions", payload, headers, self.timeout_s)
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:                     # pragma: no cover - network
            raise LLMError(f"unexpected response shape: {str(body)[:300]}") from exc
        return LLMResponse(text=text.strip(), llm_calls=1, latency_ms=watch.elapsed_ms, model=self.name)


class OllamaLLM:
    """Local Ollama daemon: ``ollama run llama3.1:8b``."""

    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, seed: int = 20260921, timeout_s: float = 120.0) -> None:
        self.name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.timeout_s = float(timeout_s)

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: Optional[Sequence[str]] = None) -> LLMResponse:
        payload = {
            "model": self.name, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_predict": int(max_tokens), **({"stop": list(stop)} if stop else {})},
        }
        with Stopwatch() as watch:
            body = _HTTPBackend.post(f"{self.base_url}/api/generate", payload,
                                     {"Content-Type": "application/json"}, self.timeout_s)
        return LLMResponse(text=str(body.get("response", "")).strip(), llm_calls=1,
                           latency_ms=watch.elapsed_ms, model=self.name)


class HFLocalLLM:
    """In-process transformers generation (greedy)."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "auto",
                 dtype: str = "auto", seed: int = 20260921) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("HFLocalLLM needs torch + transformers: pip install 'trace-rag[models]'") from exc
        from ..embeddings.hf_embedder import resolve_device

        device = resolve_device(device, torch)
        torch.manual_seed(seed)
        self.name = model_name
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if dtype == "auto":
            resolved_dtype = (torch.bfloat16 if str(device).startswith(("cuda", "xla"))
                              else torch.float32)
        else:
            resolved_dtype = getattr(torch, dtype, torch.float32)
        # device_map= needs the `accelerate` package and is only useful for
        # sharding across devices; a plain device string loads and moves the
        # model with torch alone, so a normal single-GPU or CPU run needs no
        # extra dependency.
        kwargs = {"device_map": device} if device in {"auto", "balanced", "sequential"} else {}
        try:                                         # transformers v5 renamed torch_dtype -> dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=resolved_dtype, **kwargs)
        except TypeError:  # pragma: no cover - older transformers
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=resolved_dtype, **kwargs)
        if not kwargs:
            self.model = self.model.to(device)
        self.model = self.model.eval()
        self.device = device

    def generate(self, prompt: str, max_tokens: int = 256,
                 stop: Optional[Sequence[str]] = None) -> LLMResponse:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with Stopwatch() as watch, torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
            if str(self.device).startswith("xla"):
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            generated_ids = output[0][inputs["input_ids"].shape[1]:].cpu().tolist()
            completion = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip()
        for marker in stop or ():
            if marker in completion:
                completion = completion.split(marker)[0].strip()
        return LLMResponse(text=completion, llm_calls=1, latency_ms=watch.elapsed_ms, model=self.name)


def build_llm(config):  # type: ignore[no-untyped-def]
    """Factory from ``GenerationConfig``."""
    backend = config.backend
    if backend == "stub":
        return StubLLM()
    if backend == "openai":
        return OpenAICompatLLM(config.model_name, config.base_url, config.api_key_env,
                               config.temperature, config.seed, config.timeout_s)
    if backend == "ollama":
        return OllamaLLM(config.model_name, temperature=config.temperature, seed=config.seed,
                         timeout_s=config.timeout_s)
    if backend == "huggingface":
        return HFLocalLLM(config.model_name, device=config.device, seed=config.seed)
    raise ValueError(f"unknown generation backend: {backend}")
