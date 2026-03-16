"""
ReAct agent loop: Thought -> Action -> Observation -> repeat.
"""

from typing import Any, Callable, List, Optional

from .parser import (
    format_tool_result,
    has_tool_call,
    parse_all_tool_calls,
)
from .tools import ToolRegistry


class ReActAgent:
    """
    ReAct-style agent loop.
    Uses model to generate thought + tool_call, executes tools, feeds observation back.
    """

    def __init__(
        self,
        generate_fn: Callable[[str, int, float], str],
        tools: Optional[ToolRegistry] = None,
        max_turns: int = 5,
        max_new_tokens_per_turn: int = 256,
        temperature: float = 0.7,
    ):
        """
        Args:
            generate_fn: (prompt_text, max_new_tokens, temperature) -> generated_text
            tools: Tool registry
        """
        self.generate_fn = generate_fn
        self.tools = tools or self._default_tools()
        self.max_turns = max_turns
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.temperature = temperature

    def _default_tools(self) -> ToolRegistry:
        from .tools import get_default_tools
        return get_default_tools()

    def run(self, prompt: str) -> str:
        """
        Run the agent loop until no more tool calls or max_turns.
        Returns the final response text.
        """
        tools_desc = self.tools.format_for_prompt()
        context = f"{prompt}\n\n{tools_desc}\n\nRespond with <thought>...</thought> and <tool_call>tool(args)</tool_call> when needed.\n\n"

        for turn in range(self.max_turns):
            full_prompt = context
            output_text = self.generate_fn(
                full_prompt,
                self.max_new_tokens_per_turn,
                self.temperature,
            )

            if not has_tool_call(output_text):
                return output_text.strip()

            # Execute tool calls
            calls = parse_all_tool_calls(output_text)
            observations = []
            for call in calls:
                result = self.tools.execute(call.name, **call.args)
                observations.append(format_tool_result(str(result)))

            # Append observations for next turn
            obs_text = "\n".join(observations)
            context = full_prompt + output_text + "\n" + obs_text + "\n\n"

        return "[Max turns reached]"


def create_agent_from_engine(engine, tokenizer=None) -> ReActAgent:
    """Create ReActAgent from InferenceEngine."""

    def generate_fn(prompt_text: str, max_new: int, temp: float) -> str:
        if tokenizer:
            prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
        else:
            prompt_ids = [min(ord(c), 31999) for c in prompt_text[:1024]]
        result_ids = engine.complete(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new,
            temperature=temp,
        )
        if tokenizer:
            return tokenizer.decode(result_ids)
        return "".join(chr(i) for i in result_ids if i < 65536)

    return ReActAgent(generate_fn=generate_fn)
