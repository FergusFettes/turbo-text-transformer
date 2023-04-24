import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from langchain.llms import OpenAI
from tttp.prompter import Prompter

from ttt.config import Config
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

    @staticmethod
    @click.command()
    @click.option("--display", "-d", is_flag=True, help="Display the tree")
    @click.option("--display_all", "-a", is_flag=True, help="Display all the trees")
    @click.option("--new", "-n", is_flag=True, help="Create a new root")
    @click.option("--tag", "-t", help="Tag the current tree")
    @click.option("--checkout", "-c", help="Checkout a tree by tag or index")
    @click.option("--export", "-x", help="Export to a file", default="export.md")
    def tree(display, display_all, new, tag, checkout, export):
        """Tree command"""
        config = Config.check_config()
        store = Store(config=config)
        tree = store.load_file()

        if tag:
            click.echo(f"Tagging tree with {tag}")
            tree.index.tag(tag)

        if checkout:
            click.echo(f"Checking out tree with {checkout}")
            if checkout.isdigit():
                checkout = int(checkout)
            tree.index.checkout(checkout)

        if new:
            click.echo("Creating a new root (a new tree)")
            tree.index.clear_checkout()

        if tag or checkout or new:
            tree.save()

        if display or display_all:
            if display:
                click.echo(tree.index.index_struct)
            elif display_all:
                click.echo(tree.index.index_struct.get_full_repr())

        if export:
            with open(export, "w") as fi:
                fi.write(str(tree.index))

    @staticmethod
    @click.command()
    def repl():
        config = Config.check_config()
        params = Config.load_openai_config()["engine_params"]
        store = Store(config=config)
        tree = store.load_file()
        tree.params = params

        while True:
            command = input(">> ")
            if command == "q":
                break

            if command in "hjkl":
                tree.index.step(command)
                command = "d"

            if command == "d":
                click.echo(tree.index.index_struct)
                continue

            if command == "a":
                click.echo(tree.index.index_struct.get_full_repr())
                continue

            if command == "n":
                tree.index.clear_checkout()
                continue

            if command == "p save":
                config = Config.load_openai_config()
                config["engine_params"] = tree.params
                Config.save_openai_config(config)
                click.echo("Saved.")
                continue

            if command.startswith("p "):
                name, value = command.split(" ")[1:]
                tree.params.update({name: value})
                command = "p"

            if command == "p":
                click.echo(tree.params)
                continue

            if command.startswith("c "):
                name, value = command.split(" ")[1:]
                config.update({name: value})
                command = "c"

            if command == "c":
                click.echo(config)
                continue

            if command == "context":
                click.echo(tree.index.context)
                continue

            if command.startswith("tag"):
                tree.index.tag(command.split(" ")[1])
                continue

            if command.startswith("checkout"):
                checkout = command.split(" ")[1]
                if checkout.isdigit():
                    checkout = int(checkout)
                tree.index.checkout(checkout)
                continue

            if command == "show":
                click.echo(tree.index.short_path())
                continue

            if command == "show all":
                click.echo(tree.prompt)
                continue

            if command.startswith("context add"):
                # If no context is given, just add the last node as context
                if command == "context add":
                    new_context = tree.index.path[-1].text
                else:
                    new_context = command.split(" ")[2:]
                    if Path(new_context[0]).exists():
                        new_context = Path(new_context[0]).read_text()

                tree.index.add_context(new_context, tree.index.path[-1])
                continue

            if command == "summary":
                click.echo(tree.index.last_summary)
                continue

            # Finally, assume the context is a prompt and pass it in
            prompt = command
            prompter = store.get_prompter()
            if len(tree) == 0 and prompter is not None:
                prompt = prompter.prompt(prompt)

            tree.input(prompt)
            response = tree.prompt if tree.params["model"] == "test" else simple_gen(tree)

            click.echo(response)
            tree.output(response)


def simple_gen(tree):
    encoding = Config.get_encoding(tree.params.get("model", "gpt-3.5-turbo"))
    tree.params["max_tokens"] -= len(encoding.encode(tree.prompt))
    if tree.params["model"] == "code-davinci-002":
        tree.params["openai_api_base"] = os.environ.get("CD2_URL")
        tree.params["openai_api_key"] = os.environ.get("CD2_KEY")
    llm = OpenAI(**tree.params)
    responses = llm.generate([tree.prompt])
    return responses.generations[0][0].text
