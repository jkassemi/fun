# Copyright James Kassemi <james@kassemi.org> 2025 US

# ---- 

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

from rich.text import Text
from friendly_names import generate as friendly_name
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree, TextArea
from textual.widgets.tree import TreeNode
from textual.containers import Horizontal
from textual import log
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    name: str = field(default_factory=friendly_name)
    version: int = 0
    parent: Message | None = None
    content: str | None = None


@dataclass
class System(Message):
    pass


@dataclass
class User(Message):
    pass


@dataclass
class Model(Message):
    pass


def message_label(message: Message) -> str:
    desc = message.name

    if message.content:
        desc = f"{desc} ({len(message.content)})"

    match message:
        case System():
            return f"(system) {desc}"
        case User():
            return f"(user) {desc}"
        case Model():
            return f"(model) {desc}"
        case _:
            raise ValueError(f"Unknown message type: {type(message)}")


class TreeApp(App):
    BINDINGS = [
        ("a", "add", "Add"),
        ("d", "delete", "Delete"),
        ("s", "save", "Save"),
    ]

    CSS = """
        Screen {
            layout: horizontal;
        }
    """

    def __init__(self):
        self.root = System(content="<<system>>")

        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Horizontal():
            yield Tree(message_label(self.root), data=self.root, allow_expand=False, expand=True)
            yield TextArea(text=self.root.content)

    def on_tree_node_selected(self, event: Tree.NodeSelected[Message]):
        tree = event.control
        node = event.node
        text_area = self.query_one(TextArea)
        text_area.load_text(node.data.content or "")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes with debounced updates."""
        # we don't have a selected item until this is initialized here
        tree = self.query_one(Tree)
        node = tree.cursor_node

        if not node:
            return

        text_area = event.text_area
        node.data.content = event.text_area.text

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[Message]):
        logger.info("node expanded", event)

    def action_add(self) -> None:
        """Add a node to the tree, then select it."""
        tree = self.query_one(Tree)
        node = tree.cursor_node

        match node.data:
            case System():
                new_data = User(content="")
            case User():
                new_data = Model(content="")
            case Model():
                new_data = User(content="")

        new_node = tree.cursor_node.add(
            message_label(new_data), data=new_data, allow_expand=False, expand=True
        )
        tree.select_node(node)

    def action_clear(self) -> None:
        """Clear the tree (remove all nodes)."""
        tree = self.query_one(Tree)
        tree.clear()


if __name__ == "__main__":
    logging.basicConfig(
        filename="out.log",
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
    )
    app = TreeApp()
    app.run()
