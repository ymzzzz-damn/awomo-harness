# -*- coding: utf-8 -*-
"""AgentScope chat-model adapter for the OpenAI Responses API.

Some OpenAI-compatible gateways reject ``reasoning_effort`` together with
function tools on ``/v1/chat/completions`` while supporting that combination
on ``/v1/responses``.  This adapter keeps AgentScope's chat-model contract and
translates the formatted chat history to Responses API input items.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, Type

from agentscope.model import ChatModelBase
from agentscope.model._model_response import ChatResponse
from agentscope.model._model_usage import ChatUsage
from openai import AsyncOpenAI
from pydantic import BaseModel


def _json_object(raw: Any) -> dict[str, Any]:
    """Return a tool argument object without letting malformed JSON crash."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _text_content(content: Any) -> str:
    """Flatten a chat-completions tool result to a Responses output string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {
                "text",
                "input_text",
                "output_text",
            }:
                parts.append(str(block.get("text") or ""))
            elif block is not None:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _message_content(role: str, content: Any) -> str | list[dict[str, Any]]:
    """Convert chat-completions message content to Responses message content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    converted: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            text = str(block.get("text") or "")
            if text:
                converted.append(
                    {
                        "type": "output_text" if role == "assistant" else "input_text",
                        "text": text,
                    },
                )
        elif block_type in {"image_url", "input_image"}:
            image = block.get("image_url")
            if isinstance(image, dict):
                image_url = image.get("url")
                detail = image.get("detail")
            else:
                image_url = image
                detail = block.get("detail")
            if image_url:
                image_block: dict[str, Any] = {
                    "type": "input_image",
                    "image_url": image_url,
                }
                if detail:
                    image_block["detail"] = detail
                converted.append(image_block)
    return converted


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI chat messages into Responses API input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _text_content(message.get("content")),
                    },
                )
            continue

        content = _message_content(role, message.get("content"))
        if content:
            items.append({"role": role, "content": content})

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            call_id = tool_call.get("id")
            name = function.get("name")
            if call_id and name:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": function.get("arguments") or "{}",
                    },
                )
    return items


def _responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Flatten chat-completions function schemas for the Responses API."""
    converted = []
    for tool in tools or []:
        if tool.get("type") != "function":
            converted.append(tool)
            continue
        function = tool.get("function") or {}
        entry = {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description") or "",
            "parameters": function.get("parameters")
            or {"type": "object", "properties": {}},
        }
        if "strict" in function:
            entry["strict"] = function["strict"]
        converted.append(entry)
    return converted


def _responses_tool_choice(tool_choice: Any) -> Any:
    """Translate a named chat-completions tool choice when necessary."""
    if not isinstance(tool_choice, dict):
        return tool_choice
    function = tool_choice.get("function") or {}
    name = function.get("name")
    if tool_choice.get("type") == "function" and name:
        return {"type": "function", "name": name}
    return tool_choice


class OpenAIResponsesModelCompat(ChatModelBase):
    """Non-streaming Responses API model with AgentScope content blocks."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_name=model_name, stream=False)
        self.client = AsyncOpenAI(
            api_key=api_key,
            **(client_kwargs or {}),
        )
        self.generate_kwargs = generate_kwargs or {}

    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        if structured_model is not None:
            raise NotImplementedError(
                "structured_model is not supported by the Responses adapter",
            )

        request: dict[str, Any] = {
            "model": self.model_name,
            "input": _responses_input(messages),
            **self.generate_kwargs,
            **kwargs,
        }

        effort = request.pop("reasoning_effort", None)
        if effort and effort != "none":
            reasoning = request.get("reasoning")
            if not isinstance(reasoning, dict):
                reasoning = {}
            reasoning.setdefault("effort", effort)
            # Summaries are displayable, unlike hidden chain-of-thought.
            reasoning.setdefault("summary", "auto")
            request["reasoning"] = reasoning

        if "max_tokens" in request and "max_output_tokens" not in request:
            request["max_output_tokens"] = request.pop("max_tokens")

        if tools:
            request["tools"] = _responses_tools(tools)
        if tool_choice:
            request["tool_choice"] = _responses_tool_choice(tool_choice)

        started = datetime.now()
        response = await self.client.responses.create(**request)

        content: list[dict[str, Any]] = []
        metadata = None
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                summary = "\n".join(
                    str(getattr(part, "text", "") or "")
                    for part in (getattr(item, "summary", None) or [])
                    if getattr(part, "text", None)
                ).strip()
                if summary:
                    content.append({"type": "thinking", "thinking": summary})
            elif item_type == "message":
                for part in getattr(item, "content", None) or []:
                    text = getattr(part, "text", None)
                    if text:
                        content.append({"type": "text", "text": text})
            elif item_type == "function_call":
                raw_arguments = getattr(item, "arguments", None) or "{}"
                content.append(
                    {
                        "type": "tool_use",
                        "id": getattr(item, "call_id", None)
                        or getattr(item, "id", None),
                        "name": getattr(item, "name", ""),
                        "input": _json_object(raw_arguments),
                        "raw_input": raw_arguments,
                    },
                )

        usage = None
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            usage = ChatUsage(
                input_tokens=getattr(response_usage, "input_tokens", 0) or 0,
                output_tokens=getattr(response_usage, "output_tokens", 0) or 0,
                time=(datetime.now() - started).total_seconds(),
                metadata=response_usage,
            )

        return ChatResponse(
            id=getattr(response, "id", None) or "",
            content=content,
            usage=usage,
            metadata=metadata,
        )


__all__ = [
    "OpenAIResponsesModelCompat",
    "_responses_input",
    "_responses_tools",
]
