from dataclasses import dataclass
from typing import Protocol

from agent_runner.mcps.catalog import ResolvedMcpServer


@dataclass(frozen=True)
class McpPolicyDecision:
    """Decision returned after evaluating one MCP server Tool against platform policy."""

    allowed: bool
    """Whether the Tool may be executed."""

    reason: str = ""
    """Credential-safe explanation when execution is denied."""


class McpExecutionPolicy(Protocol):
    def evaluate(self, server: ResolvedMcpServer, tool_name: str) -> McpPolicyDecision:
        """Evaluate platform disables before server execution policy.

        Args:
            server: Resolved or connected MCP server participating in the operation.
            tool_name: Provider-visible Tool name.

        Returns:
            Evaluated platform disables before server execution policy.
        """


class FullAccessServerPolicy:
    def evaluate(self, server: ResolvedMcpServer, tool_name: str) -> McpPolicyDecision:
        """Reject administrator-disabled servers or Tools before remote execution.

        Args:
            server: Resolved server profile and effective credential snapshot.
            tool_name: Provider-visible Tool name requested by the model.

        Returns:
            An allow decision or a bounded reason suitable for diagnostics.
        """

        if not server.profile.enabled:
            return McpPolicyDecision(False, "MCP server is disabled")

        if tool_name in server.profile.disabled_tools:
            return McpPolicyDecision(False, "MCP tool is disabled")

        return McpPolicyDecision(True)
