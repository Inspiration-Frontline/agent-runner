import ast
import operator
from collections.abc import Callable

from agent_runner.tools.registry import BaseTool


class CalculatorTool(BaseTool):
    _binary_operators: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    _unary_operators: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @property
    def tool_key(self) -> str:
        return "builtin.calculator"

    @property
    def tool_name(self) -> str:
        return "calculate_expression"

    @property
    def description(self) -> str:
        return "Evaluate a code-style arithmetic expression using +, -, *, /, parentheses, and decimal numbers."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression such as (12.5 + 7.5) / 4.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    @property
    def strict(self) -> bool:
        return True

    async def execute(self, expression: str) -> dict:
        normalized = expression.strip()
        if not normalized:
            raise ValueError("Expression cannot be blank.")
        if len(normalized) > 1_000:
            raise ValueError("Expression is too long.")

        try:
            parsed = ast.parse(normalized, mode="eval")
            value = self._evaluate(parsed.body, depth=0)
        except (SyntaxError, TypeError) as error:
            raise ValueError("Expression contains unsupported syntax.") from error
        if not isinstance(value, int | float):
            raise ValueError("Expression did not produce a number.")
        return {"expression": normalized, "result": value}

    def _evaluate(self, node: ast.AST, depth: int) -> int | float:
        if depth > 64:
            raise ValueError("Expression nesting is too deep.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary_operators:
            left = self._evaluate(node.left, depth + 1)
            right = self._evaluate(node.right, depth + 1)
            return self._binary_operators[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_operators:
            return self._unary_operators[type(node.op)](self._evaluate(node.operand, depth + 1))
        # TODO: Extend the safe code-style grammar with functions, constants, powers, and other
        # advanced expressions without accepting LaTeX or evaluating arbitrary Python code.
        raise ValueError("Expression contains unsupported syntax.")
