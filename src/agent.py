"""Generic agent loop using OpenAI function calling."""

import json
import time
from typing import Callable
from openai import OpenAI
from src.tools import TOOL_DISPATCH

MODEL = "gpt-4.1-mini"
MAX_TURNS = 10

# Callback signature: (event, data) where event is one of:
#   "tool_call_start" -> data = {"name": str, "arguments": dict, "timestamp": float}
#   "tool_call_end"   -> data = {"name": str, "arguments": dict, "timestamp": float, "duration": float}
OnToolCall = Callable[[str, dict], None]


def run_agent(
    client: OpenAI,
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_dispatch: dict | None = None,
    model: str = MODEL,
    max_turns: int = MAX_TURNS,
    messages: list[dict] | None = None,
    response_format: type | None = None,
    on_tool_call: OnToolCall | None = None,
) -> tuple[str, list[dict]]:
    """Run an agent loop with tool calling.

    Returns (final_response_text, full_messages).
    """
    if tool_dispatch is None:
        tool_dispatch = TOOL_DISPATCH

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    for _ in range(max_turns):
        kwargs = {"model": model, "messages": messages, "tools": tools}
        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # Append assistant message
        messages.append(msg.model_dump(exclude_none=True))

        # If no tool calls, we're done
        if not msg.tool_calls:
            return msg.content or "", messages

        # Execute tool calls
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if on_tool_call:
                on_tool_call("tool_call_start", {
                    "name": fn_name,
                    "arguments": fn_args,
                    "timestamp": time.time(),
                })

            t0 = time.time()
            if fn_name in tool_dispatch:
                result = tool_dispatch[fn_name](fn_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            if on_tool_call:
                on_tool_call("tool_call_end", {
                    "name": fn_name,
                    "arguments": fn_args,
                    "timestamp": time.time(),
                    "duration": time.time() - t0,
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # If we hit max turns, return whatever we have
    return messages[-1].get("content", "Max turns reached"), messages
