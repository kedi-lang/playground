from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
from pydantic_core import to_jsonable_python

_AGENT_NAME = "kedi-webgpu"


class WebGPUTelemetry:
    """Record browser-hosted model runs using OpenTelemetry GenAI conventions."""

    def __init__(self, tracer: Tracer, *, run_id: str, model: str) -> None:
        self._tracer = tracer
        self._run_id = run_id
        self._model = model
        self._all_messages: list[dict[str, Any]] = []
        self._system_instructions: list[dict[str, str]] = []
        self._input_tokens = 0
        self._output_tokens = 0

    @contextmanager
    def run_span(self) -> Iterator[Callable[[Any], None]]:
        attributes = {
            "model_name": self._model,
            "agent_name": _AGENT_NAME,
            "gen_ai.agent.name": _AGENT_NAME,
            "gen_ai.agent.call.id": self._run_id,
            "gen_ai.conversation.id": self._run_id,
            "gen_ai.operation.name": "invoke_agent",
            "logfire.msg": f"{_AGENT_NAME} run",
        }
        with self._tracer.start_as_current_span(
            f"invoke_agent {_AGENT_NAME}",
            attributes=attributes,
        ) as span:
            yield lambda result: self._finish_run_span(span, result)

    @contextmanager
    def model_request_span(
        self,
        payload: Mapping[str, Any],
    ) -> Iterator[Callable[[Mapping[str, Any]], None]]:
        model = str(payload.get("model", self._model))
        attributes = self._model_request_attributes(payload, model)
        with self._tracer.start_as_current_span(
            f"chat {model}",
            attributes=attributes,
            kind=SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield lambda response: self._finish_model_span(span, payload, response, model)
            except BaseException as exc:
                span.record_exception(exc, escaped=True)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    @contextmanager
    def tool_span(
        self,
        name: str,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> Iterator[Callable[[Any], None]]:
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.id": call_id,
            "gen_ai.tool.call.arguments": _json(arguments),
            "gen_ai.agent.name": _AGENT_NAME,
            "gen_ai.agent.call.id": self._run_id,
            "gen_ai.conversation.id": self._run_id,
            "logfire.msg": f"running tool: {name}",
            "logfire.json_schema": _json(
                {
                    "type": "object",
                    "properties": {
                        "gen_ai.tool.call.arguments": {"type": "object"},
                        "gen_ai.tool.call.result": {"type": "object"},
                        "gen_ai.tool.name": {},
                        "gen_ai.tool.call.id": {},
                    },
                }
            ),
        }
        with self._tracer.start_as_current_span(
            f"execute_tool {name}",
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield lambda result: span.set_attribute(
                    "gen_ai.tool.call.result",
                    _json(result),
                )
            except BaseException as exc:
                span.record_exception(exc, escaped=True)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def _model_request_attributes(
        self,
        payload: Mapping[str, Any],
        model: str,
    ) -> dict[str, Any]:
        tools = _tool_definitions(payload.get("tools"))
        parameters = {
            "function_tools": [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters_json_schema": tool.get("parameters", {}),
                }
                for tool in tools
            ],
            "output_schema": payload.get("outputSchema"),
            "required_tools": payload.get("requiredTools", []),
        }
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "webgpu",
            "gen_ai.system": "webgpu",
            "gen_ai.request.model": model,
            "gen_ai.agent.name": _AGENT_NAME,
            "gen_ai.agent.call.id": self._run_id,
            "gen_ai.conversation.id": self._run_id,
            "model_request_parameters": _json(parameters),
            "kedi.webgpu.step": int(payload.get("step", 0)),
            "logfire.msg": f"chat {model}",
            "logfire.json_schema": _json(
                {
                    "type": "object",
                    "properties": {
                        "model_request_parameters": {"type": "object"},
                        "gen_ai.input.messages": {"type": "array"},
                        "gen_ai.output.messages": {"type": "array"},
                        "gen_ai.system_instructions": {"type": "array"},
                        "gen_ai.tool.definitions": {"type": "array"},
                    },
                }
            ),
        }
        if tools:
            attributes["gen_ai.tool.definitions"] = _json(tools)
        settings = payload.get("settings")
        if isinstance(settings, Mapping):
            for key in (
                "max_tokens",
                "top_p",
                "seed",
                "temperature",
            ):
                value = settings.get(key)
                if isinstance(value, int | float):
                    attributes[f"gen_ai.request.{key}"] = value
        return attributes

    def _finish_model_span(
        self,
        span: Span,
        payload: Mapping[str, Any],
        response: Mapping[str, Any],
        request_model: str,
    ) -> None:
        telemetry = response.get("telemetry")
        telemetry_map = telemetry if isinstance(telemetry, Mapping) else {}
        raw_messages = telemetry_map.get("inputMessages", payload.get("messages", []))
        input_messages = _otel_messages(raw_messages)
        output_messages = [_otel_output_message(response, telemetry_map)]
        instructions = _system_instructions(raw_messages, payload.get("instructions"))

        span.set_attributes(
            {
                "gen_ai.input.messages": _json(input_messages),
                "gen_ai.output.messages": _json(output_messages),
                "gen_ai.response.model": str(telemetry_map.get("model") or request_model),
            }
        )
        if instructions:
            span.set_attribute("gen_ai.system_instructions", _json(instructions))
        response_id = telemetry_map.get("responseId")
        if isinstance(response_id, str) and response_id:
            span.set_attribute("gen_ai.response.id", response_id)
        finish_reason = telemetry_map.get("finishReason")
        if not isinstance(finish_reason, str) or not finish_reason:
            finish_reason = "tool_call" if response.get("kind") == "tool_call" else "stop"
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])

        usage = _usage(telemetry_map.get("usage"))
        if usage["input_tokens"]:
            span.set_attribute("gen_ai.usage.input_tokens", usage["input_tokens"])
        if usage["output_tokens"]:
            span.set_attribute("gen_ai.usage.output_tokens", usage["output_tokens"])
        self._input_tokens += usage["input_tokens"]
        self._output_tokens += usage["output_tokens"]
        self._all_messages = [*input_messages, *output_messages]
        self._system_instructions = instructions

        if response.get("kind") == "error":
            error = RuntimeError(str(response.get("error") or "Browser model request failed"))
            span.record_exception(error, escaped=False)
            span.set_status(Status(StatusCode.ERROR, str(error)))

    def _finish_run_span(self, span: Span, result: Any) -> None:
        span.set_attribute(
            "final_result",
            result if isinstance(result, str) else _json(result),
        )
        span.set_attribute("pydantic_ai.all_messages", _json(self._all_messages))
        if self._system_instructions:
            span.set_attribute(
                "gen_ai.system_instructions",
                _json(self._system_instructions),
            )
        if self._input_tokens:
            span.set_attribute(
                "gen_ai.aggregated_usage.input_tokens",
                self._input_tokens,
            )
        if self._output_tokens:
            span.set_attribute(
                "gen_ai.aggregated_usage.output_tokens",
                self._output_tokens,
            )
        span.set_attribute(
            "logfire.json_schema",
            _json(
                {
                    "type": "object",
                    "properties": {
                        "final_result": {"type": "object"},
                        "pydantic_ai.all_messages": {"type": "array"},
                        "gen_ai.system_instructions": {"type": "array"},
                    },
                }
            ),
        )


def _tool_definitions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    definitions: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            continue
        definition: dict[str, Any] = {
            "type": "function",
            "name": item["name"],
        }
        description = item.get("description")
        if isinstance(description, str) and description:
            definition["description"] = description
        parameters = item.get("inputSchema")
        if isinstance(parameters, Mapping):
            definition["parameters"] = dict(parameters)
        definitions.append(definition)
    return definitions


def _otel_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        messages.append(_otel_message(role, content, item))
    return messages


def _otel_message(
    role: str,
    content: str,
    message: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = _parse_json(content)
    if (
        role == "assistant"
        and isinstance(parsed, Mapping)
        and (parsed.get("type") == "tool_call" or parsed.get("action") == "call_tool")
    ):
        return {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": str(parsed.get("callId") or ""),
                    "name": str(parsed.get("name") or ""),
                    "arguments": parsed.get("arguments", {}),
                }
            ],
        }
    if role == "tool":
        result = parsed if parsed is not None else content
        return {
            "role": "user",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": str(message.get("toolCallId") or ""),
                    "name": str(message.get("name") or ""),
                    "result": result,
                }
            ],
        }
    return {
        "role": role if role in {"user", "system", "assistant"} else "user",
        "parts": [{"type": "text", "content": content}],
    }


def _otel_output_message(
    response: Mapping[str, Any],
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    finish_reason = telemetry.get("finishReason")
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "tool_call" if response.get("kind") == "tool_call" else "stop"
    if response.get("kind") == "tool_call":
        parts = [
            {
                "type": "tool_call",
                "id": str(response.get("callId") or ""),
                "name": str(response.get("name") or ""),
                "arguments": response.get("arguments", {}),
            }
        ]
    else:
        output_text = telemetry.get("outputText")
        if not isinstance(output_text, str):
            output_text = _json(response.get("data"))
        parts = [{"type": "text", "content": output_text}]
    return {
        "role": "assistant",
        "parts": parts,
        "finish_reason": finish_reason,
    }


def _system_instructions(raw_messages: Any, fallback: Any) -> list[dict[str, str]]:
    if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, str | bytes):
        for item in raw_messages:
            if (
                isinstance(item, Mapping)
                and item.get("role") == "system"
                and isinstance(item.get("content"), str)
            ):
                return [{"type": "text", "content": item["content"]}]
    if isinstance(fallback, str) and fallback:
        return [{"type": "text", "content": fallback}]
    return []


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": _nonnegative_int(value.get("inputTokens", value.get("prompt_tokens", 0))),
        "output_tokens": _nonnegative_int(
            value.get("outputTokens", value.get("completion_tokens", 0))
        ),
    }


def _nonnegative_int(value: Any) -> int:
    if not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _json(value: Any) -> str:
    return json.dumps(
        to_jsonable_python(value, fallback=str),
        ensure_ascii=False,
    )


__all__ = ["WebGPUTelemetry"]
