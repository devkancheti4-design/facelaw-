"""Evaluate an authored lane exactly as compiled C (int32, wrap-around): every operation is wrapped to 32 bits."""
import ast

_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.BitAnd: lambda a, b: a & b,
        ast.BitOr: lambda a, b: a | b, ast.BitXor: lambda a, b: a ^ b, ast.LShift: lambda a, b: a << b,
        ast.RShift: lambda a, b: a >> b, ast.Mult: lambda a, b: a * b}


def s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def _ev(n, x):
    if isinstance(n, ast.Constant):
        return s32(n.value)
    if isinstance(n, ast.Name):
        return x
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, x)
        return s32(-v) if isinstance(n.op, ast.USub) else s32(~v)
    if isinstance(n, ast.BinOp):
        return s32(_BIN[type(n.op)](_ev(n.left, x), _ev(n.right, x)))
    raise ValueError("not a lane expression: %s" % ast.dump(n))


_CACHE = {}


def E(expr, x):
    t = _CACHE.get(expr)
    if t is None:
        t = _CACHE[expr] = ast.parse(expr, mode="eval").body
    return _ev(t, s32(int(x)))
