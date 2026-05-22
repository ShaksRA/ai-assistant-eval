"""
Tool use for the Frontier assistant (Claude Sonnet).
Implements: calculator, current datetime, simple unit converter.

These are passed as Anthropic tool_use blocks and executed locally.
"""
import math
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression. "
            "Supports basic arithmetic, powers, sqrt, trig, log, etc. "
            "Input must be a safe Python expression."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time in UTC.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone_offset_hours": {
                    "type": "number",
                    "description": "Optional UTC offset in hours (e.g. 5.5 for IST, -5 for EST)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "unit_converter",
        "description": "Convert between common units (length, weight, temperature, speed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'km', 'kg', 'celsius'"},
                "to_unit": {"type": "string", "description": "Target unit, e.g. 'miles', 'lbs', 'fahrenheit'"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
]

# ---------------------------------------------------------------------------
# Safe math evaluator
# ---------------------------------------------------------------------------

_SAFE_MATH_GLOBALS = {
    "__builtins__": {},
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": math.pow, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
    "floor": math.floor, "ceil": math.ceil,
}


def _calculate(expression: str) -> str:
    try:
        result = eval(expression, _SAFE_MATH_GLOBALS, {})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

CONVERSIONS = {
    # Length
    ("km", "miles"): lambda v: v * 0.621371,
    ("miles", "km"): lambda v: v * 1.60934,
    ("m", "ft"): lambda v: v * 3.28084,
    ("ft", "m"): lambda v: v * 0.3048,
    ("cm", "inches"): lambda v: v * 0.393701,
    ("inches", "cm"): lambda v: v * 2.54,
    # Weight
    ("kg", "lbs"): lambda v: v * 2.20462,
    ("lbs", "kg"): lambda v: v * 0.453592,
    ("g", "oz"): lambda v: v * 0.035274,
    ("oz", "g"): lambda v: v * 28.3495,
    # Temperature
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    ("celsius", "kelvin"): lambda v: v + 273.15,
    ("kelvin", "celsius"): lambda v: v - 273.15,
    # Speed
    ("kmh", "mph"): lambda v: v * 0.621371,
    ("mph", "kmh"): lambda v: v * 1.60934,
    ("ms", "kmh"): lambda v: v * 3.6,
    ("kmh", "ms"): lambda v: v / 3.6,
}


def _convert_units(value: float, from_unit: str, to_unit: str) -> str:
    key = (from_unit.lower().strip(), to_unit.lower().strip())
    if key in CONVERSIONS:
        result = CONVERSIONS[key](value)
        return f"{value} {from_unit} = {result:.4g} {to_unit}"
    return f"Conversion from '{from_unit}' to '{to_unit}' is not supported."


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_input: Dict) -> str:
    """Execute a tool call and return the string result."""
    if tool_name == "calculator":
        return _calculate(tool_input["expression"])

    elif tool_name == "get_current_datetime":
        offset = tool_input.get("timezone_offset_hours", 0)
        from datetime import timedelta
        now = datetime.now(timezone.utc) + timedelta(hours=offset)
        tz_label = f"UTC{'+' if offset >= 0 else ''}{offset}" if offset else "UTC"
        return f"{now.strftime('%A, %B %d, %Y %H:%M:%S')} {tz_label}"

    elif tool_name == "unit_converter":
        return _convert_units(
            tool_input["value"],
            tool_input["from_unit"],
            tool_input["to_unit"],
        )

    return f"Unknown tool: {tool_name}"
