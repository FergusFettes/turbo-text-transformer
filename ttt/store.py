import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click


@dataclass
class Store:
    """
    Indexs are stored in a vector db, which is a collection of documents

    This handles the vector db, plus exporting/importing to disk.
    """

    chat_path: Path = Path("~/.config/ttt/chats/").expanduser()
    template_path: Path = Path("~/.config/ttt/templates/").expanduser()
    config: Optional["Config"] = None  # type: ignore

    def __post_init__(self):
        if self.config and self.config["chat_path"]:
            self.dir = Path(self.config["chat_path"]).expanduser()
        if self.config and self.config["template_path"]:
            self.template_dir = Path(self.config["template_path"]).expanduser()

    def list_chats(self):
        if self.chat_path.exists():
            self._list_dir()

    def _list_dir(self):
        files = [x for x in self.chat_path.glob("*.json")]
        click.echo(f"Found {len(files)} chats.")
        summaries = [json.loads(x.read_text()).get("summary", None) for x in files]
        for file, summary in zip(files, summaries):
            summary = summary or "No summary"
            click.echo(f"{file.stem}: {summary:100}")

    def list_templates(self):
        if not self.template_path.exists():
            return

        files = [x for x in self.template_path.glob("*.j2")]
        click.echo(f"Found {len(files)} templates.")
        for file in files:
            lines = file.read_text().splitlines()
            click.echo(f"{file.stem}:\t\t{lines[0]}")
