"""Calculator skill for Fairy 2.0. Safe math evaluator + unit conversions."""
import ast
import json
import math
import operator
import re
from typing import Any

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs, "round": round, "sqrt": math.sqrt, "ceil": math.ceil,
    "floor": math.floor, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "degrees": math.degrees, "radians": math.radians, "hypot": math.hypot,
    "factorial": math.factorial, "gcd": math.gcd, "pow": pow,
    "min": min, "max": max, "pi": math.pi, "e": math.e, "tau": math.tau,
}

def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op}")
        return _SAFE_OPS[op](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary: {op}")
        return _SAFE_OPS[op](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        fname = node.func.id
        if fname not in _SAFE_FUNCS:
            raise ValueError(f"Function not allowed: {fname}")
        return _SAFE_FUNCS[fname](*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> str:
    expr = str(expression).strip()
    expr = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
    expr = re.sub(r'\bx\b', '*', expr)
    if any(kw in expr for kw in ('import', 'exec', 'eval', '__', 'open')):
        return json.dumps({"error": "Expression contains disallowed keywords."})
    try:
        tree = ast.parse(expr, mode='eval')
        result = _safe_eval(tree)
    except ZeroDivisionError:
        return json.dumps({"error": "Division by zero."})
    except (ValueError, TypeError, OverflowError, SyntaxError) as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Calculation failed: {exc}"})

    if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
        formatted = str(int(result))
    else:
        formatted = f"{result:.10g}"
    return json.dumps({"expression": expression, "result": result, "formatted": formatted})


_CONVERSIONS = {
    ("c","f"): lambda v: v*9/5+32, ("f","c"): lambda v: (v-32)*5/9,
    ("c","k"): lambda v: v+273.15, ("k","c"): lambda v: v-273.15,
    ("km","mi"): lambda v: v*0.621371, ("mi","km"): lambda v: v*1.60934,
    ("m","ft"): lambda v: v*3.28084, ("ft","m"): lambda v: v*0.3048,
    ("cm","in"): lambda v: v*0.393701, ("in","cm"): lambda v: v*2.54,
    ("kg","lb"): lambda v: v*2.20462, ("lb","kg"): lambda v: v*0.453592,
    ("g","oz"): lambda v: v*0.035274, ("oz","g"): lambda v: v*28.3495,
    ("l","gal"): lambda v: v*0.264172, ("gal","l"): lambda v: v*3.78541,
    ("kmh","mph"): lambda v: v*0.621371, ("mph","kmh"): lambda v: v*1.60934,
}

def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    f, t = from_unit.lower().strip(), to_unit.lower().strip()
    fn = _CONVERSIONS.get((f, t))
    if not fn:
        return json.dumps({"error": f"Conversion '{from_unit}' to '{to_unit}' not supported."})
    try:
        result = fn(float(value))
        return json.dumps({"input": f"{value} {from_unit}", "result": round(result, 8),
                           "formatted": f"{value} {from_unit} = {result:.6g} {to_unit}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def run(**kwargs) -> str:
    expression = kwargs.get("expression") or kwargs.get("expr") or kwargs.get("query")
    value = kwargs.get("value")
    from_unit = kwargs.get("from_unit") or kwargs.get("from")
    to_unit = kwargs.get("to_unit") or kwargs.get("to")
    if value is not None and from_unit and to_unit:
        return convert_units(float(value), str(from_unit), str(to_unit))
    if expression:
        return calculate(str(expression))
    return json.dumps({"error": "Provide 'expression' or 'value + from_unit + to_unit'."})


class Calculator:
    calculate = staticmethod(calculate)
    convert = staticmethod(convert_units)
