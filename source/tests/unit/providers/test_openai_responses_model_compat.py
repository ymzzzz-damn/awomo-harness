# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

from qwenpaw.providers.openai_responses_model_compat import (
    OpenAIResponsesModelCompat,
    _responses_input,
    _responses_tools,
)


def test_responses_input_preserves_images_and_tool_round_trip() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,abc",
                        "detail": "high",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "status", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "ok",
        },
    ]

    assert _responses_input(messages) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "inspect"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc",
                    "detail": "high",
                },
            ],
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "status",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "ok",
        },
    ]


def test_responses_tools_are_flattened() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "status",
                "description": "Read status",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        },
    ]
    assert _responses_tools(tools) == [
        {
            "type": "function",
            "name": "status",
            "description": "Read status",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        },
    ]


async def test_model_maps_reasoning_text_and_function_call(monkeypatch) -> None:
    captured = {}
    response = SimpleNamespace(
        id="resp-1",
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text="Checked the request.")],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call-1",
                id="fc-1",
                name="status",
                arguments='{"detail":true}',
            ),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
    )

    class FakeResponses:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return response

    model = OpenAIResponsesModelCompat(
        model_name="gpt-test",
        api_key="test",
        generate_kwargs={"reasoning_effort": "high"},
    )
    model.client = SimpleNamespace(responses=FakeResponses())

    result = await model(
        messages=[{"role": "user", "content": "check"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "status",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        tool_choice="required",
    )

    assert captured["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "reasoning_effort" not in captured
    assert result.content[0] == {
        "type": "thinking",
        "thinking": "Checked the request.",
    }
    assert result.content[1]["type"] == "tool_use"
    assert result.content[1]["input"] == {"detail": True}
    assert result.usage.input_tokens == 10
