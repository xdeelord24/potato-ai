"""Quick test of the agent loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.parser import parse_tool_call, has_tool_call
from agent.tools import get_default_tools


def main():
    # Test parser
    text = '<thought>I need to search</thought><tool_call>search(query="hello")</tool_call>'
    assert has_tool_call(text)
    call = parse_tool_call(text)
    assert call and call.name == "search" and call.args.get("query") == "hello"
    print("Parser: OK")

    # Test tools
    tools = get_default_tools()
    r = tools.execute("search", query="test")
    assert "Mock" in r
    r = tools.execute("get_weather", location="Tokyo")
    assert "Tokyo" in r
    print("Tools: OK")

    # Test agent (with mock generate)
    from agent.loop import ReActAgent

    def mock_gen(prompt, max_new, temp):
        return "The answer is 42."

    agent = ReActAgent(generate_fn=mock_gen, max_turns=2)
    out = agent.run("What is 2+2?")
    assert "42" in out or "answer" in out.lower()
    print("Agent: OK")
    print("All agent tests passed.")


if __name__ == "__main__":
    main()
