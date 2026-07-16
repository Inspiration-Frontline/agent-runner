from agent_runner.tools.internal.calculator import CalculatorTool
from agent_runner.tools.internal.current_time import CurrentTimeTool
from agent_runner.tools.internal.weather import WeatherTool
from agent_runner.tools.internal.web_search import WebSearchTool
from agent_runner.tools.registry import ToolRegistry


def build_internal_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (CurrentTimeTool(), CalculatorTool(), WeatherTool(), WebSearchTool()):
        registry.register(tool.to_definition())
    return registry
