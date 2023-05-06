import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ttt.config import Config, click, shell
from ttt.tree import DummyTree, Tree


@dataclass
class Store:
    """
    Indexs are stored in a vector db, which is a collection of documents

    This handles the vector db, plus exporting/importing to disk.
    """

    config: Optional["Config"] = None  # type: ignore

    chat_path: Path = Path("~/.config/ttt/chats/").expanduser()

    def __post_init__(self):
        if self.config and self.config.get("chat_path", None):
            self.chat_path = Path(self.config["chat_path"]).expanduser()
            self.chat_file = self.chat_path / f"{Path(self.config['chat_name'])}.json"

    def dump(self):
        if self.chat_file.exists():
            return json.dumps(json.loads(self.chat_file.read_text()), indent=4)
        return json.dumps({})

    def list_files(self):
        if self.chat_path.exists():
            self._list_dir()

    def _list_dir(self):
        files = [x for x in self.chat_path.glob("*.json")]
        click.echo(f"Found {len(files)} chats.")
        summaries = [json.loads(x.read_text()).get("summary", None) for x in files]
        for file, summary in zip(files, summaries):
            summary = summary or "No summary"
            click.echo(f"{file.stem}: {summary:100}")

    def list_db_docs(self):
        if not self.db:
            return

        click.echo(f"Found {len(self.db.documents)} documents.")
        for doc in self.db.documents:
            click.echo(f"{doc['name']}: {doc['summary']}")

    def load_file(self):
        if self.config["file"]:
            return Tree(self.chat_file)
        return DummyTree()

    @staticmethod
    @click.command()
    @click.option("--toggle_file", "-t", help="Set file mode on/off", is_flag=True, default=False)
    @click.option("--default_file", "-d", help="Set default file.", default="")
    @click.option("--list", "-l", help="List available files.", is_flag=True, default=False)
    @click.option("--dump", "-D", help="Dump default chat file.", is_flag=True, default=False)
    def file(toggle_file, default_file, list, dump):
        config = Config.check_config()
        Config.check_file(toggle_file, default_file, config)
        if list:
            Store(config=config).list_files()
        if dump:
            click.echo(Store(config=config).dump())
