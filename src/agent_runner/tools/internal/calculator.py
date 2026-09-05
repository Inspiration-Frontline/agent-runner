import ast
import operator
from collections.abc import Callable
from types import MappingProxyType

from agents import function_tool

_BINARY_OPERATORS: MappingProxyType[type[ast.operator], Callable[[float, float], float]] = MappingProxyType(
    {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
)
_UNARY_OPERATORS: MappingProxyType[type[ast.unaryop], Callable[[float], float]] = MappingProxyType(
    {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
)


@function_tool(failure_error_function=None)
async def calculate_expression(expression: str) -> dict[str, str | int | float]:
    """Evaluate an arithmetic expression using +, -, *, /, parentheses, and decimal numbers.

    Args:
        expression: Code-style arithmetic expression such as (12.5 + 7.5) / 4.

    Returns:
        Evaluated an arithmetic expression using +, -, *, /, parentheses, and decimal numbers.
    """
    normalized: str = expression.strip()

    if not normalized:
        raise ValueError("Expression cannot be blank.")

    if len(normalized) > 1_000:
        raise ValueError("Expression is too long.")

    try:
        parsed: ast.Expression = ast.parse(normalized, mode="eval")
        value: int | float = _evaluate(parsed.body, depth=0)
    except (SyntaxError, TypeError) as error:
        raise ValueError("Expression contains unsupported syntax.") from error

    if not isinstance(value, int | float):
        raise ValueError("Expression did not produce a number.")

    return {"expression": normalized, "result": value}


def _evaluate(node: ast.AST, depth: int) -> int | float:
    """Evaluate the restricted arithmetic AST without executing arbitrary Python code.

    Args:
        node: Restricted arithmetic AST node to evaluate.
        depth: Current recursion depth used to enforce the safety limit.

    Returns:
        Evaluated the restricted arithmetic AST without executing arbitrary Python code.
    """

    if depth > 64:
        raise ValueError("Expression nesting is too deep.")

    if isinstance(node, ast.Constant):
        value: object = node.value

        if isinstance(value, int | float) and not isinstance(value, bool):
            return value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left: int | float = _evaluate(node.left, depth + 1)
        right: int | float = _evaluate(node.right, depth + 1)

        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, depth + 1))
    # TODO: Extend the safe code-style grammar with functions, constants, powers, and other
    # advanced expressions without accepting LaTeX or evaluating arbitrary Python code.
    raise ValueError("Expression contains unsupported syntax.")
