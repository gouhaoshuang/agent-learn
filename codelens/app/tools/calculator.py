"""Safe calculator tool for numeric expressions.

This module intentionally does not use ``eval``. It parses the expression with
``ast`` and evaluates only a small whitelist of numeric syntax and math
functions.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Callable

from langchain_core.tools import tool


MAX_EXPR_LENGTH = 200
MAX_ABS_NUMBER = 1e100
MAX_POWER_EXPONENT = 100

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MATH_FUNCS: dict[str, Callable[..., float]] = {
    name: getattr(math, name)
    for name in (
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "cos",
        "degrees",
        "exp",
        "fabs",
        "floor",
        "hypot",
        "log",
        "log10",
        "log2",
        "pow",
        "radians",
        "sin",
        "sqrt",
        "tan",
    )
}

_CONSTANTS = {
    "e": math.e,
    "pi": math.pi,
    "tau": math.tau,
}


class CalculatorError(ValueError):
    """User-facing calculator error."""


def _ensure_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("only numeric values are allowed")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("result is not finite")
    if abs(value) > MAX_ABS_NUMBER:
        raise CalculatorError("result is too large")
    return value


class _Evaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError("only numeric literals are allowed")
        return _ensure_number(node.value)

    def visit_Name(self, node: ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"unknown name: {node.id}")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError("unsupported unary operator")
        return _ensure_number(op(self.visit(node.operand)))

    def visit_BinOp(self, node: ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError("unsupported binary operator")

        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise CalculatorError(
                f"exponent too large; maximum absolute exponent is {MAX_POWER_EXPONENT}"
            )

        try:
            return _ensure_number(op(left, right))
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("result overflowed") from None

    def visit_Call(self, node: ast.Call):
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("only direct math function calls are allowed")

        func = _MATH_FUNCS.get(node.func.id)
        if func is None:
            raise CalculatorError(f"unsupported function: {node.func.id}")

        args = [self.visit(arg) for arg in node.args]
        try:
            return _ensure_number(func(*args))
        except (TypeError, ValueError) as exc:
            raise CalculatorError(str(exc)) from None
        except OverflowError:
            raise CalculatorError("result overflowed") from None

    def generic_visit(self, node: ast.AST):
        raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def _format_result(value) -> str:
    value = _ensure_number(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".12g")
    return str(value)


def calculate_expression(expression: str) -> str:
    """Safely evaluate a numeric expression and return the result as text."""
    expr = (expression or "").strip()
    if not expr:
        return "(error) expression must not be empty"
    if len(expr) > MAX_EXPR_LENGTH:
        return f"(error) expression is too long; max length is {MAX_EXPR_LENGTH}"

    try:
        tree = ast.parse(expr, mode="eval")
        result = _Evaluator().visit(tree)
        return _format_result(result)
    except SyntaxError as exc:
        return f"(error) invalid expression: {exc.msg}"
    except CalculatorError as exc:
        return f"(error) {exc}"


@tool
def calculator(expression: str) -> str:
    """Calculate a numeric expression safely.

    Supports numbers, parentheses, +, -, *, /, //, %, **, unary +/- and selected
    math functions/constants such as sqrt, sin, cos, log, pi, e, and tau. Does
    not execute Python code.
    """
    return calculate_expression(expression)
