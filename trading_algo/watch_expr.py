"""Safe AST-based expression evaluator for `watch --until`.

Agents type `watch quote --until "last > 150"` — we must not `eval()`
that. Instead, parse into a restricted AST and walk it ourselves.

Allowed:
- Comparisons: `<, <=, ==, !=, >=, >` (chained OK)
- Logical: `and, or, not`
- Arithmetic: `+, -, *, /, %`
- Names (resolved against the current snapshot dict)
- Number / string / bool / None literals

Disallowed (AST-rejected at parse time):
- function calls
- attribute access (a.b)
- subscripts (a[b])
- comprehensions, lambdas, assignments, import
"""

from __future__ import annotations

import ast
import operator
from typing import Any


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}


class UnsafeExpression(ValueError):
    """Raised when a disallowed node type appears in the expression."""


def _eval_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        # Missing keys resolve to None — makes the expression tolerant to
        # fields the broker sometimes omits.
        return None
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"unary {type(node.op).__name__} not allowed")
        return op(_eval_node(node.operand, env))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"binary {type(node.op).__name__} not allowed")
        return op(_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise UnsafeExpression(f"bool op {type(node.op).__name__} not allowed")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            fn = _CMP_OPS.get(type(op))
            if fn is None:
                raise UnsafeExpression(f"comparator {type(op).__name__} not allowed")
            right = _eval_node(comp, env)
            if left is None or right is None:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    result = fn(left, right)
                else:
                    return False
            else:
                result = fn(left, right)
            if not result:
                return False
            left = right
        return True
    raise UnsafeExpression(f"{type(node).__name__} not allowed in watch expression")


def evaluate(expr: str, snapshot: dict[str, Any]) -> bool:
    """Evaluate `expr` against `snapshot`. Unknown names resolve to None."""
    if not expr or not expr.strip():
        raise ValueError("empty expression")
    tree = ast.parse(expr, mode="eval")
    return bool(_eval_node(tree, snapshot))
