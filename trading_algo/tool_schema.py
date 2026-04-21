"""JSONSchema generation for every CLI subcommand.

`tools-describe` emits an array of tool specs ready to paste into a
Claude `tools` parameter or a GPT function-call spec. Schema is derived
by introspecting live argparse — when flags change, schemas auto-update.

Output shape (per subcommand):

    {
      "name": "place-order",
      "description": "Place a single test order",
      "input_schema": {
        "type": "object",
        "properties": {...},
        "required": [...]
      },
      "output_schema": {...},
      "examples": [...]
    }
"""

from __future__ import annotations

import argparse
from typing import Any


def _arg_to_jsonschema(action: argparse.Action) -> dict[str, Any]:
    prop: dict[str, Any] = {}

    if action.help:
        # Argparse help strings escape '%' as '%%' internally — un-escape
        # before emitting so the JSONSchema description is clean.
        prop["description"] = action.help.replace("%%", "%")

    if action.choices:
        prop["enum"] = list(action.choices)

    if isinstance(action, argparse._StoreTrueAction):
        prop["type"] = "boolean"
        prop["default"] = False
    elif isinstance(action, argparse._StoreFalseAction):
        prop["type"] = "boolean"
        prop["default"] = True
    else:
        py_type = action.type or str
        if py_type is int:
            prop["type"] = "integer"
        elif py_type is float:
            prop["type"] = "number"
        elif py_type is bool:
            prop["type"] = "boolean"
        else:
            prop["type"] = "string"

    if action.default is not None and "default" not in prop:
        if not isinstance(action, argparse._StoreTrueAction):
            # Defaults that aren't JSON-safe (e.g. argparse SUPPRESS) get
            # stringified; the agent still sees something useful.
            try:
                import json
                json.dumps(action.default)
                prop["default"] = action.default
            except (TypeError, ValueError):
                prop["default"] = str(action.default)

    return prop


def _flag_to_property_name(opt_string: str) -> str:
    """`--good-till-date` → `good_till_date`."""
    return opt_string.lstrip("-").replace("-", "_")


def _subparser_schema(subparser: argparse.ArgumentParser) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for action in subparser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if action.dest in ("cmd", "func"):
            continue

        if not action.option_strings:
            if action.default is argparse.SUPPRESS:
                continue
            properties[action.dest] = _arg_to_jsonschema(action)
            if action.required or action.default is None:
                required.append(action.dest)
            continue

        long_flag = max(action.option_strings, key=len)
        name = _flag_to_property_name(long_flag)
        properties[name] = _arg_to_jsonschema(action)
        if action.required:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = sorted(required)
    return schema


def _output_schema_for(cmd_name: str) -> dict[str, Any]:
    """Canonical envelope output schema."""
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "cmd": {"const": cmd_name},
            "schema_version": {"type": "string"},
            "request_id": {"type": "string"},
            "data": {},
            "warnings": {"type": "array", "items": {"type": "object"}},
            "meta": {"type": "object"},
            "error": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "class": {"type": "string"},
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean"},
                    "ib_error_code": {"type": "integer"},
                    "field_errors": {"type": "array"},
                    "suggested_action": {"type": "string"},
                },
            },
        },
        "required": ["ok", "cmd", "schema_version", "request_id"],
    }


def _examples_for(cmd_name: str) -> list[dict]:
    """Pull up to 3 example notes from the explain module."""
    try:
        from trading_algo.explain import all_explanations
    except Exception:
        return []
    body = all_explanations().get(cmd_name, {})
    notes = body.get("notes", [])
    return [{"description": n} for n in notes[:3]]


def describe_tools(parser: argparse.ArgumentParser) -> list[dict]:
    """Return the full tool-spec array for every subcommand in `parser`.

    Iterates `_SubParsersAction` and synthesises one entry per subcommand.
    Result is JSON-serialisable and sorted by name.
    """
    sub_action = None
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            sub_action = a
            break
    if sub_action is None:
        return []

    tools: list[dict] = []
    for name, sub in sub_action.choices.items():
        desc = sub.description or (sub.format_usage() or "").strip()
        spec: dict[str, Any] = {
            "name": name,
            "description": desc,
            "input_schema": _subparser_schema(sub),
            "output_schema": _output_schema_for(name),
        }
        ex = _examples_for(name)
        if ex:
            spec["examples"] = ex
        tools.append(spec)
    tools.sort(key=lambda t: t["name"])
    return tools
