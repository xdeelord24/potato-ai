"""
Parse model output for tool calls in ReAct format.
Format: <thought>...</thought><tool_call>tool_name(args)</tool_call>
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ToolCall:
    """Parsed tool invocation."""

    name: str
    args: dict


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Extract a single tool call from model output.
    Supports: tool_name(arg="value") or tool_name(key=123)
    """
    match = re.search(
        r"<tool_call>\s*(\w+)\s*\((.*?)\)\s*</tool_call>",
        text,
        re.DOTALL,
    )
    if not match:
        return None

    name = match.group(1)
    args_str = match.group(2).strip()

    args = {}
    if args_str:
        # Parse key="value" or key=value
        for part in re.findall(r'(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*(\S+)', args_str):
            if part[0]:
                args[part[0]] = part[1]
            else:
                args[part[2]] = part[3].strip("'\"")

    return ToolCall(name=name, args=args)


def parse_thought(text: str) -> Optional[str]:
    """Extract thought content."""
    match = re.search(r"<thought>\s*(.*?)\s*</thought>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_all_tool_calls(text: str) -> List[ToolCall]:
    """Extract all tool calls from text."""
    calls = []
    for match in re.finditer(
        r"<tool_call>\s*(\w+)\s*\((.*?)\)\s*</tool_call>",
        text,
        re.DOTALL,
    ):
        name = match.group(1)
        args_str = match.group(2).strip()
        args = {}
        if args_str:
            for part in re.findall(r'(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*(\S+)', args_str):
                if part[0]:
                    args[part[0]] = part[1]
                else:
                    args[part[2]] = part[3].strip("'\"")
        calls.append(ToolCall(name=name, args=args))
    return calls


def has_tool_call(text: str) -> bool:
    """Check if text contains a tool call."""
    return "<tool_call>" in text and "</tool_call>" in text


def format_tool_result(observation: str) -> str:
    """Wrap observation in XML for model consumption."""
    return f"<observation>\n{observation}\n</observation>"
