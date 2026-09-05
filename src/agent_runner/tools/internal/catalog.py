from agents import FunctionTool

from agent_runner.tools.internal.calculator import calculate_expression
from agent_runner.tools.internal.current_time import get_current_time
from agent_runner.tools.internal.weather import get_current_weather
from agent_runner.tools.internal.web_search import search_web
from agent_runner.tools.registry import ToolDefinition, ToolRegistry

INTERNAL_FUNCTION_TOOLS: tuple[tuple[str, FunctionTool], ...] = (
    ("builtin.current_time", get_current_time),
    ("builtin.calculator", calculate_expression),
    ("builtin.weather", get_current_weather),
    ("builtin.web_search", search_web),
)


def build_internal_tool_registry() -> ToolRegistry:
    """Build the request-independent registry of approved built-in Tools.

    Returns:
        build the request-independent registry of approved built-in Tools.
    """
    registry: ToolRegistry = ToolRegistry()

    for tool_key, function_tool in INTERNAL_FUNCTION_TOOLS:
        registry.register(ToolDefinition.from_function_tool(tool_key, function_tool))

    return registry
