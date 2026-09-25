"""An offline, deterministic "chat model" so the example runs with no API key.

It answers extractively: it picks the context sentence(s) that best overlap the question.
That is crude, but it fails in realistic ways (weak retrieval, obeying injected text), which
is exactly what makes it a good demo for an evaluation tool.

* ``acme-extractive-v1`` returns one sentence, ``acme-extractive-v2`` returns two.
* If the system prompt calls the context *untrusted*, sentences that look like injected
  instructions are dropped - a toy stand-in for a hardened prompt.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from examples.rag_agent.knowledge_base import tokenize

ABSTAIN = "There is not enough information in the knowledge base to answer that."
_INJECTION = re.compile(r"ignore (?:all )?(?:your |previous )|system override|disregard", re.I)


class ExtractiveChatModel(BaseChatModel):
    model_name: str = "acme-extractive-v1"
    min_overlap: float = 0.34
    token_delay_s: float = 0.0005

    @property
    def _llm_type(self) -> str:
        return "acme-extractive"

    def _answer(self, messages: list[BaseMessage]) -> str:
        system = " ".join(str(m.content) for m in messages if m.type == "system")
        prompt = str(messages[-1].content)
        calc = re.search(r"^Calculation: (.+)$", prompt, re.M)
        if calc:
            return calc.group(1).strip()
        context_part, _, question = prompt.partition("Question:")
        question = question.strip().splitlines()[0] if question.strip() else ""
        q_tokens = set(tokenize(question))
        sentences: list[str] = []
        for line in context_part.splitlines():
            line = re.sub(r"^\[\d+\]\s*", "", line.strip())
            if line and not line.startswith("Context"):
                sentences.extend(s for s in re.split(r"(?<=[.!?])\s+", line) if s)
        if "untrusted" in system.lower():
            sentences = [s for s in sentences if not _INJECTION.search(s)]
        scored = []
        for i, sentence in enumerate(sentences):
            s_tokens = set(tokenize(sentence))
            overlap = len(q_tokens & s_tokens)
            # more matched words wins; ties go to the more focused (shorter) sentence
            scored.append((overlap, overlap / (len(s_tokens) or 1), -i, sentence))
        scored.sort(reverse=True)
        best = [s for s in scored if q_tokens and s[0] / len(q_tokens) >= self.min_overlap]
        if not best:
            return ABSTAIN
        n = 2 if self.model_name.endswith("v2") else 1
        chosen = sorted(best[:n], key=lambda s: -s[2])
        return " ".join(s[3] for s in chosen)

    def _usage(self, messages: list[BaseMessage], text: str) -> dict[str, int]:
        n_in = sum(len(str(m.content).split()) for m in messages)
        n_out = len(text.split())
        return {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._answer(messages)
        time.sleep(self.token_delay_s * (4 + len(text.split())))
        message = AIMessage(
            content=text,
            usage_metadata=self._usage(messages, text),  # type: ignore[arg-type]
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        text = self._answer(messages)
        time.sleep(self.token_delay_s * 4)  # "time to first token"
        words = text.split(" ")
        for i, word in enumerate(words):
            piece = word + (" " if i < len(words) - 1 else "")
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            time.sleep(self.token_delay_s)
            yield chunk
        final = AIMessageChunk(
            content="",
            usage_metadata=self._usage(messages, text),  # type: ignore[arg-type]
            response_metadata={"model_name": self.model_name},
        )
        yield ChatGenerationChunk(message=final)
