"""Print the instrument reference (the content of docs/instruments.md)."""
from sparta.instruments import render_markdown

print(render_markdown(), end="")
