"""Sentences the server fills in but the UI translates."""

from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class Sentence(str):
    """An English sentence that keeps its catalog key: ``template``, with
    ``{{name}}`` placeholders, and the ``params`` that fill them. It is the
    sentence wherever a string is (logs, ``str()``, JSON); the UI translates
    ``template`` with ``params`` (``localizations/zh_CN.json``), which a
    sentence formatted with an f-string could never match. Used for a rejected
    value's message (``ParamError``), model advice and why a script does not
    apply.

    A param is text the server does not translate (a value, a model or file
    name), a list of names the catalogs have (capability labels): the UI
    translates each and joins them its own way; the English joins them with
    ", ", or a sentence of its own (why a field of an object was refused),
    which the UI translates from its template and params."""

    template: str
    params: dict[str, str | list[str]]

    def __new__(cls, template: str, /, **params: Any) -> Sentence:
        values: dict[str, str | list[str]] = {
            name: [item if isinstance(item, Sentence) else str(item) for item in value] if isinstance(value, (list, tuple)) else value if isinstance(value, Sentence) else str(value)
            for name, value in params.items()
        }
        english = {name: ", ".join(value) if isinstance(value, list) else value for name, value in values.items()}
        sentence = super().__new__(cls, _PLACEHOLDER.sub(lambda match: english.get(match.group(1), match.group(0)), template))
        sentence.template = template
        sentence.params = values
        return sentence

    def __getnewargs_ex__(self) -> tuple[tuple[str], dict[str, str | list[str]]]:
        # copy and pickle rebuild a sentence from these: from its English text, a "{{word}}" a
        # param put there was read as a placeholder of its own
        return (self.template,), {name: list(value) if isinstance(value, list) else value for name, value in self.params.items()}


def sentence_fields(text: Any, prefix: str = "") -> dict[str, Any]:
    """``{"template", "params"}`` (names prefixed) of a :class:`Sentence` with
    params, else ``{}``: what a JSON object adds next to the English text."""

    if isinstance(text, Sentence) and text.params:
        return {f"{prefix}template": text.template, f"{prefix}params": _json_params(text.params)}
    return {}


def error_fields(text: Any) -> dict[str, Any]:
    """``{"template", "params"}`` of a :class:`Sentence`, also of one without
    params, else ``{}``: what the body of a refused request adds next to its
    English ``detail``. The template tells the UI the detail is a sentence its
    catalogs translate; a detail without one is said by its status."""

    return {"template": text.template, "params": _json_params(text.params)} if isinstance(text, Sentence) else {}


def _json_params(params: dict[str, str | list[str]], prefix: str = "") -> dict[str, str | list[str]]:
    """Params as JSON. A sentence param is its English text, with its template and params
    beside it under its name (``reason.template``, ``reason.value``): still all strings, and
    no ``{{placeholder}}`` can be named so (a name with a dot is not a word)."""

    result: dict[str, str | list[str]] = {}
    for name, value in params.items():
        if isinstance(value, list):
            result[prefix + name] = [str(item) for item in value]
            for index, item in enumerate(value):
                if isinstance(item, Sentence):
                    result[f"{prefix}{name}.{index}.template"] = item.template
                    result.update(_json_params(item.params, f"{prefix}{name}.{index}."))
        elif isinstance(value, Sentence):
            result[prefix + name] = str(value)
            result[f"{prefix}{name}.template"] = value.template
            result.update(_json_params(value.params, f"{prefix}{name}."))
        else:
            result[prefix + name] = value
    return result
