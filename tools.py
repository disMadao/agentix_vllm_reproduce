from __future__ import annotations

import ast
import operator
from datetime import datetime


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_expr(tree.body))
    except Exception as exc:
        return f"calculator error: {exc}"


def lookup_stub(query: str) -> str:
    return f"lookup_stub: 没有联网检索，收到查询：{query}"


def time_stub(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def echo(text: str) -> str:
    return text


TOOLS = {
    "calculator": calculator,
    "lookup_stub": lookup_stub,
    "time_stub": time_stub,
    "echo": echo,
}


def _eval_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("only numeric expressions are allowed")
