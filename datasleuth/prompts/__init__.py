"""Versioned prompt templates.

Prompts live in ``prompts/<version>/<name>.md``. Placeholders are ``{{name}}`` and are
substituted verbatim (no format-string parsing, so JSON braces are safe).
"""

from __future__ import annotations

from functools import cache
from importlib import resources

PROMPT_VERSION = "v1"


@cache
def _load(version: str, name: str) -> str:
    return (
        resources.files("datasleuth.prompts")
        .joinpath(version, f"{name}.md")
        .read_text(encoding="utf-8")
        .strip()
    )


def render(name: str, /, **values: str) -> str:
    """Load prompt ``name`` for the active version and substitute ``{{key}}`` placeholders."""
    text = _load(PROMPT_VERSION, name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def data_block(tag: str, content: str) -> str:
    """Wrap untrusted content in a delimited block the model is told never to obey."""
    safe = content.replace(f"</{tag}>", f"< /{tag}>")
    return f"<{tag}>\n{safe}\n</{tag}>"
