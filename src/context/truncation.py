import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    max_tokens: int
    used_tokens: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def can_fit(self, tokens: int) -> bool:
        return self.used_tokens + tokens <= self.max_tokens

    def use(self, tokens: int):
        self.used_tokens += tokens


class TokenBudgetManager:
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens

    def create_budget(self) -> TokenBudget:
        return TokenBudget(max_tokens=self.max_tokens)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def truncate_messages(
        self,
        messages: list[dict],
        budget: TokenBudget,
        truncation_strategy: str = "sliding_window",
    ) -> list[dict]:
        if truncation_strategy == "sliding_window":
            return self._sliding_window_truncation(messages, budget)
        elif truncation_strategy == "importance":
            return self._importance_truncation(messages, budget)
        else:
            return self._sliding_window_truncation(messages, budget)

    def _sliding_window_truncation(self, messages: list[dict], budget: TokenBudget) -> list[dict]:
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
        return self._sliding_window_truncation(messages, budget)
