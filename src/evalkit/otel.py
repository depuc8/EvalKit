"""Export traces as OpenTelemetry spans (``pip install evalkit-agents[otel]``)."""

from __future__ import annotations

from typing import Any

from evalkit.trace import Step, Trace


class OTelExporter:
    """Callable ``(Trace) -> None`` that replays a finished trace as spans.

    Spans carry the original start/end times, so latency in your tracing backend matches the
    recording. Prompts, completions and tool arguments are only attached when
    ``capture_content=True`` (they may contain sensitive data).

    Pass it anywhere a trace exporter is accepted (``run_eval(..., exporters=[OTelExporter()])``).
    """

    def __init__(
        self,
        tracer_provider: Any = None,
        *,
        tracer_name: str = "evalkit",
        capture_content: bool = False,
    ) -> None:
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:
            raise ImportError(
                "opentelemetry is required: pip install 'evalkit-agents[otel]'"
            ) from exc
        self._otel = otel_trace
        provider = tracer_provider or otel_trace.get_tracer_provider()
        self._tracer = provider.get_tracer(tracer_name)
        self.capture_content = capture_content

    def __call__(self, trace: Trace) -> None:
        from opentelemetry.trace import Status, StatusCode

        ns = 1_000_000_000
        start = int(trace.start_time * ns)
        root = self._tracer.start_span(
            f"agent.run {trace.agent}",
            start_time=start,
            attributes={
                "evalkit.trace_id": trace.trace_id,
                "evalkit.case_id": trace.case_id or "",
                "evalkit.agent": trace.agent,
            },
        )
        if trace.error:
            root.set_status(Status(StatusCode.ERROR, trace.error))
        spans: dict[str, Any] = {}
        pending = list(trace.steps)
        while pending:
            progressed = False
            for step in list(pending):
                if (
                    step.parent_id is not None
                    and step.parent_id not in spans
                    and any(p.id == step.parent_id for p in pending)
                ):
                    continue  # wait until the parent span exists
                pending.remove(step)
                progressed = True
                parent = spans.get(step.parent_id or "", root)
                span = self._tracer.start_span(
                    f"{step.kind} {step.name}",
                    context=self._otel.set_span_in_context(parent),
                    start_time=int(step.start_time * ns),
                    attributes=self._attributes(step),
                )
                if step.error:
                    span.set_status(Status(StatusCode.ERROR, step.error))
                span.end(end_time=int((step.start_time + step.latency_s) * ns))
                spans[step.id] = span
            if not progressed:  # pragma: no cover - guards against parent cycles
                break
        root.end(end_time=start + int(trace.latency_s * ns))

    def _attributes(self, step: Step) -> dict[str, Any]:
        attrs: dict[str, Any] = {"evalkit.step.kind": step.kind, "evalkit.step.id": step.id}
        if step.llm is not None:
            attrs["gen_ai.request.model"] = step.llm.model or ""
            attrs["gen_ai.usage.input_tokens"] = step.llm.usage.input_tokens
            attrs["gen_ai.usage.output_tokens"] = step.llm.usage.output_tokens
            if step.llm.time_to_first_token_s is not None:
                attrs["evalkit.llm.time_to_first_token_s"] = step.llm.time_to_first_token_s
            if self.capture_content:
                attrs["gen_ai.prompt"] = step.llm.prompt_text
                attrs["gen_ai.completion"] = step.llm.output
        if step.tool is not None:
            attrs["tool.name"] = step.tool.name
            if self.capture_content:
                attrs["tool.args"] = repr(step.tool.args)
                attrs["tool.result"] = repr(step.tool.result)
        return attrs
