import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain.llms import OpenAI
from tttp.prompter import Prompter

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
    template_path: Path = Path("~/.config/ttt/templates/").expanduser()

    def __post_init__(self):
        if self.config and self.config.get("chat_path", None):
            self.chat_path = Path(self.config["chat_path"]).expanduser()
            self.chat_file = self.chat_path / f"{Path(self.config['chat_name'])}.json"
        if self.config and self.config.get("template_path", None):
            self.template_path = Path(self.config["template_path"]).expanduser()
            self.template_file = self.template_path / f"{Path(self.config['template_file'])}.j2"

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

    def list_templates(self):
        if not self.template_path.exists():
            return

        files = [x for x in self.template_path.glob("*.j2")]
        click.echo(f"Found {len(files)} templates.")
        for file in files:
            lines = file.read_text().splitlines()
            click.echo(f"{file.stem}:\t\t{lines[0]}")

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

    def get_prompter(self, template_file=None, template_args=None):
        if not self.config["template"]:
            return None
        if template_file:
            self.template_file = self.template_path / Path(template_file)
        if self.template_file.exists():
            return Prompter(self.template_file, args=Config.arg2dict(template_args))

    def get_tree(self, prompt, prompter, params):
        params = Config.prepare_engine_params(params)
        tree = self.load_file()
        tree.params = params

        # If its a simple gen or it is the first item in a new tree, apply the prompt
        if len(tree) == 0 and prompter is not None:
            prompt = prompter.prompt(prompt)

        tree.input(prompt)
        return tree

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

    @staticmethod
    @click.command()
    @click.option("--toggle_template", "-t", help="Set template mode on/off", is_flag=True, default=False)
    @click.option("--default_template", "-d", help="Set default template.", default="")
    @click.option("--list", "-l", help="List available templates.", is_flag=True, default=False)
    def template(toggle_template, default_template, list):
        config = Config.check_config()
        Config.check_template(toggle_template, default_template, config)
        if list:
            Store(config=config).list_templates()
