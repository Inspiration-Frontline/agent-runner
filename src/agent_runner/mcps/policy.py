from dataclasses import dataclass
from typing import Protocol

from agent_runner.mcps.catalog import ResolvedMcpServer


@dataclass(frozen=True)
class McpPolicyDecision:
    allowed: bool
    reason: str = ""


class McpExecutionPolicy(Protocol):
    def evaluate(self, server: ResolvedMcpServer, tool_name: str) -> McpPolicyDecision:
        """Evaluate platform disables before server execution policy."""


class FullAccessServerPolicy:
    def evaluate(self, server: ResolvedMcpServer, tool_name: str) -> McpPolicyDecision:
        if not server.profile.enabled:
            return McpPolicyDecision(False, "MCP server is disabled")
        if tool_name in server.profile.disabled_tools:
            return McpPolicyDecision(False, "MCP tool is disabled")
        return McpPolicyDecision(True)
