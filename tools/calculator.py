"""Safe calculator tool.

Expressions are parsed with `ast` and evaluated by walking a whitelist of node types.
There is no eval/exec, no attribute access, no names except whitelisted functions and
constants, and exponent/sequence sizes are bounded.
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import Any, Callable

from pydantic import BaseModel, Field

MAX_EXPONENT = 1000
MAX_SEQUENCE = 10_000
MAX_EXPRESSION_LENGTH = 2000


class CalculatorInput(BaseModel):
    expression: str = Field(
        description=(
            "Math expression. Operators: + - * / // % **. Functions: sqrt, log, log10, exp, abs, round, "
            "min, max, sum, mean, median, stdev, pstdev, variance, pct_of(p, total), pct(part, total), "
            "pct_change(old, new), cagr(start, end, years). Lists allowed, e.g. mean([1,2,3])."
        )
    )


class CalculatorOutput(BaseModel):
    expression: str
    result: float | list[float]
    formatted: str


def _pct_of(p: float, total: float) -> float:
    return p / 100 * total


def _pct(part: float, total: float) -> float:
    return part / total * 100


def _pct_change(old: float, new: float) -> float:
    return (new - old) / old * 100


def _cagr(start: float, end: float, years: float) -> float:
    return ((end / start) ** (1 / years) - 1) * 100


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs, "round": round,
    "min": min, "max": max, "sum": sum, "mean": statistics.fmean, "median": statistics.median,
    "stdev": statistics.stdev, "pstdev": statistics.pstdev, "variance": statistics.variance,
    "pct_of": _pct_of, "pct": _pct, "pct_change": _pct_change, "cagr": _cagr,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorError(ValueError):
    pass


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalculatorError(f"Unknown name '{node.id}'")
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_SEQUENCE:
            raise CalculatorError("Sequence too long")
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise CalculatorError("Exponent too large")
            return math.pow(float(left), float(right))  # float pow: bounded time, raises OverflowError
        return BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        return FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> CalculatorOutput:
    expr = expression.strip().replace("^", "**")
    if not expr or len(expr) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError("Expression is empty or too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {exc.msg}") from exc
    try:
        value = _eval(tree)
    except (ZeroDivisionError, OverflowError, statistics.StatisticsError, TypeError, ValueError) as exc:
        raise CalculatorError(f"{type(exc).__name__}: {exc}") from exc
    if isinstance(value, list):
        result: float | list[float] = [float(v) for v in value]
        formatted = ", ".join(f"{v:,.6g}" for v in result)
    else:
        result = float(value)
        formatted = f"{result:,.6g}"
    return CalculatorOutput(expression=expression, result=result, formatted=formatted)


def run(args: CalculatorInput, _context: Any) -> CalculatorOutput:
    return calculate(args.expression)
