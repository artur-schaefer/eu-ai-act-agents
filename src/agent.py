"""Generic agent loop using OpenAI function calling."""

import json
from openai import OpenAI
from src.tools import TOOL_DISPATCH

MODEL = "gpt-4.1-mini"
MAX_TURNS = 10


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

            if fn_name in tool_dispatch:
                result = tool_dispatch[fn_name](fn_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # If we hit max turns, return whatever we have
    return messages[-1].get("content", "Max turns reached"), messages
