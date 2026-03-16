"""
Tool registry and execution for the agent.
"""

from typing import Any, Callable, Dict, Optional


class ToolRegistry:
    """Registry of callable tools for the agent."""

    def __init__(self):
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any], description: str = "") -> None:
        """Register a tool."""
        self._tools[name] = func

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name with kwargs."""
        func = self._tools.get(name)
        if func is None:
            return f"Error: Unknown tool '{name}'"
        try:
            return func(**kwargs)
        except Exception as e:
            return f"Error: {e}"

    def list_tools(self) -> Dict[str, str]:
        """List available tools and their docstrings."""
        return {
            name: (func.__doc__ or "").strip()
            for name, func in self._tools.items()
        }

    def format_for_prompt(self) -> str:
        """Format tool list for inclusion in model prompt."""
        lines = ["Available tools:"]
        for name, func in self._tools.items():
            doc = (func.__doc__ or "No description").strip().split("\n")[0]
            lines.append(f"  - {name}: {doc}")
        return "\n".join(lines)


# Built-in tools for testing
def search(query: str) -> str:
    """Search for information. (Mock implementation)"""
    return f"[Mock search result for: {query}]"


def calculator(expression: str) -> str:
    """Evaluate a math expression. (Mock - use eval with caution in production)"""
    try:
        # Safe subset - no imports, no builtins
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(location: str) -> str:
    """Get weather for a location. (Mock implementation)"""
    return f"[Mock weather for {location}: 72°F, sunny]"


def get_default_tools() -> ToolRegistry:
    """Create registry with default mock tools."""
    reg = ToolRegistry()
    reg.register("search", search, "Search for information")
    reg.register("calculator", calculator, "Evaluate math expression")
    reg.register("get_weather", get_weather, "Get weather for location")
    return reg
