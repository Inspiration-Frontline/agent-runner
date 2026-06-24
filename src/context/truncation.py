import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """
    Token budget tracker for context management.

    Tracks the allocation and usage of tokens within a maximum budget,
    providing methods to check availability and consume tokens.

    Attributes:
        max_tokens: Maximum number of tokens allowed in this budget.
        used_tokens: Number of tokens currently used.
    """

    max_tokens: int
    used_tokens: int = 0

    @property
    def remaining_tokens(self) -> int:
        """
        Calculate the number of remaining tokens in the budget.

        Returns:
            int: Number of tokens still available for use.
        """
        return max(0, self.max_tokens - self.used_tokens)

    def can_fit(self, tokens: int) -> bool:
        """
        Check if a given number of tokens can fit within the remaining budget.

        Args:
            tokens: Number of tokens to check.

        Returns:
            bool: True if tokens can fit, False otherwise.
        """
        return self.used_tokens + tokens <= self.max_tokens

    def use(self, tokens: int):
        """
        Consume tokens from the budget.

        Args:
            tokens: Number of tokens to consume.
        """
        self.used_tokens += tokens


class TokenBudgetManager:
    """
    Manager for token budget and message truncation.

    Provides functionality to create token budgets, estimate token counts,
    and truncate messages to fit within budget constraints using various
    strategies.

    Attributes:
        max_tokens: Default maximum tokens for budgets.
    """

    def __init__(self, max_tokens: int = 128000):
        """
        Initialize the token budget manager.

        Args:
            max_tokens: Maximum number of tokens for budgets created by this manager.
        """
        self.max_tokens = max_tokens

    def create_budget(self) -> TokenBudget:
        """
        Create a new token budget with the configured maximum.

        Returns:
            TokenBudget: A new token budget instance.
        """
        return TokenBudget(max_tokens=self.max_tokens)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Uses a simple heuristic of 4 characters per token.

        Args:
            text: The text to estimate tokens for.

        Returns:
            int: Estimated number of tokens.
        """
        return len(text) // 4

    def truncate_messages(
        self,
        messages: list[dict],
        budget: TokenBudget,
        truncation_strategy: str = "sliding_window",
    ) -> list[dict]:
        """
        Truncate messages to fit within a token budget.

        Args:
            messages: List of messages to truncate.
            budget: Token budget to fit messages into.
            truncation_strategy: Strategy to use: 'sliding_window' or 'importance'.

        Returns:
            list[dict]: Truncated list of messages that fit within the budget.
        """
        if truncation_strategy == "sliding_window":
            return self._sliding_window_truncation(messages, budget)
        elif truncation_strategy == "importance":
            return self._importance_truncation(messages, budget)
        else:
            return self._sliding_window_truncation(messages, budget)

    def _sliding_window_truncation(self, messages: list[dict], budget: TokenBudget) -> list[dict]:
        """
        Truncate messages using a sliding window approach.

        Keeps the most recent messages that fit within the budget,
        discarding older messages.

        Args:
            messages: List of messages to truncate.
            budget: Token budget to fit messages into.

        Returns:
            list[dict]: Truncated list of messages.
        """
        truncated = []
        total_tokens = 0

        for message in reversed(messages):
            message_tokens = self.estimate_tokens(str(message.get("content", "")))
            if total_tokens + message_tokens <= budget.remaining_tokens:
                truncated.insert(0, message)
                total_tokens += message_tokens
            else:
                break

        return truncated

    def _importance_truncation(self, messages: list[dict], budget: TokenBudget) -> list[dict]:
        """
        Truncate messages based on importance scoring.

        Currently falls back to sliding window truncation.
        Future implementation will prioritize messages by importance.

        Args:
            messages: List of messages to truncate.
            budget: Token budget to fit messages into.

        Returns:
            list[dict]: Truncated list of messages.
        """
        return self._sliding_window_truncation(messages, budget)
