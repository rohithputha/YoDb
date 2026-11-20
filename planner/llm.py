"""Generators: how the planner gets text from a model.

* :class:`OpenAICompatibleGenerator` talks to any OpenAI-compatible chat endpoint (vLLM serving
  Qwen2.5-Coder-7B-Instruct on the research node, Ollama, an API) with JSON output and, when the
  server supports it, the output JSON Schema as a decoding grammar (vLLM's ``guided_json``).
* :class:`TransformersGenerator` runs the model in-process with Hugging Face transformers (4-bit
  when ``bitsandbytes`` is installed), greedy decoding, the schema as a grammar when ``outlines``
  is installed.
* :class:`ScriptedGenerator` returns canned answers, for tests and dry runs.

All count tokens, so the benchmark can report them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, Protocol
import urllib.request


@dataclass(frozen=True)
class Generation:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


class Generator(Protocol):
    name: str

    def generate(self, messages: Sequence[dict[str, str]], *, json_schema: dict[str, Any] | None = None) -> Generation: ...


class OpenAICompatibleGenerator:
    """``POST {base_url}/chat/completions`` with temperature 0; ``guided_json`` for vLLM, ``response_format`` otherwise."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        max_tokens: int = 1500,
        guided: bool = True,
        timeout: float = 120.0,
        thinking: bool = False,
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        self.max_tokens = max_tokens
        self.guided = guided
        self.timeout = timeout
        # A reasoning model (Qwen3, ...) may think before it answers.  Thinking is turned off with the
        # model's soft switch so the answer is the JSON alone; what it still emits in <think> tags is removed.
        self.thinking = thinking

    # How the schema is handed over, tried in this order until the server accepts one (remembered afterwards):
    # OpenAI's structured outputs, vLLM's guided decoding, plain JSON mode, nothing.
    MODES = ("json_schema", "guided_json", "json_object", "none")

    def generate(self, messages: Sequence[dict[str, str]], *, json_schema: dict[str, Any] | None = None) -> Generation:
        import urllib.error

        modes = [m for m in self.MODES if self.guided or m in ("json_object", "none")] if json_schema is not None else ["none"]
        if getattr(self, "_mode", None) in modes:
            modes = modes[modes.index(self._mode):]
        messages = list(messages)
        if not self.thinking and messages and messages[-1]["role"] == "user" and "/no_think" not in messages[-1]["content"]:
            messages[-1] = {**messages[-1], "content": messages[-1]["content"] + " /no_think"}
        last_error: Exception | None = None
        for mode in modes:
            body: dict[str, Any] = {
                "model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens, "seed": 0,
            }
            if mode == "json_schema":
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": json_schema}}
            elif mode == "guided_json":
                body["guided_json"] = json_schema
            elif mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code in (400, 422) and mode != "none":
                    last_error = error
                    continue                                    # the server does not know this mode: try the next
                raise
            self._mode = mode
            latency = (time.perf_counter() - started) * 1000
            text = strip_thinking(payload["choices"][0]["message"]["content"] or "")
            usage = payload.get("usage") or {}
            return Generation(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), latency)
        raise RuntimeError(f"the endpoint rejected every request mode: {last_error}")


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's <think>...</think> block(s); an unclosed block is removed to its end."""

    text = _THINK.sub("", text)
    if "<think>" in text:
        text = text[: text.index("<think>")]
    return text.strip()


class TransformersGenerator:
    """Qwen2.5-Coder (or any chat model) in-process; greedy; 4-bit when possible."""

    def __init__(self, model: str = "Qwen/Qwen2.5-Coder-7B-Instruct", *, load_in_4bit: bool = True, max_new_tokens: int = 1500, device_map: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            except Exception:                                   # no bitsandbytes: full precision
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.max_new_tokens = max_new_tokens
        self._outlines = None
        try:
            import outlines  # noqa: F401

            self._outlines = outlines
        except ImportError:
            pass

    def generate(self, messages: Sequence[dict[str, str]], *, json_schema: dict[str, Any] | None = None) -> Generation:
        import torch

        prompt = self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        started = time.perf_counter()
        if json_schema is not None and self._outlines is not None:
            model = self._outlines.models.Transformers(self.model, self.tokenizer)
            generator = self._outlines.generate.json(model, json.dumps(json_schema))
            text = json.dumps(generator(prompt, max_tokens=self.max_new_tokens))
            tokens_out = len(self.tokenizer(text)["input_ids"])
        else:
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = output[0][inputs["input_ids"].shape[1]:]
            text = strip_thinking(self.tokenizer.decode(generated, skip_special_tokens=True))
            tokens_out = int(generated.shape[0])
        return Generation(text, int(inputs["input_ids"].shape[1]), tokens_out, (time.perf_counter() - started) * 1000)


class ScriptedGenerator:
    """Answers from a list (one per call) or a function of the messages; counts characters/4 as tokens."""

    name = "scripted"

    def __init__(self, answers: Sequence[Any] | None = None, *, respond=None) -> None:
        self._answers = list(answers or [])
        self._respond = respond
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: Sequence[dict[str, str]], *, json_schema: dict[str, Any] | None = None) -> Generation:
        self.calls.append(list(messages))
        if self._respond is not None:
            answer = self._respond(messages)
        elif self._answers:
            answer = self._answers.pop(0)
        else:
            raise RuntimeError("the scripted generator has no answer left")
        text = answer if isinstance(answer, str) else json.dumps(answer)
        prompt_chars = sum(len(m["content"]) for m in messages)
        return Generation(text, prompt_chars // 4, len(text) // 4, 0.0)
