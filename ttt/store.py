import json
import os
import sys
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
    def exit(_, command_params, __):
        if command_params and command_params[0] == "help":
            click.echo("Quits! You can also just do Ctrl+C.")
            return
        sys.exit(0)

    @staticmethod
    def params(_, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo(
                "Show params, or edit them with 'params param=value param=value'. You must run 'params save' to persist changes."
            )
            return

        if command_params[0] == "save":
            config = Config.load_openai_config()
            config["engine_params"] = tree.params
            Config.save_openai_config(config)
            click.echo("Saved.")
            return

        if command_params:
            for command in command_params:
                name, value = command.split("=")
                tree.params.update({name: value})
        click.echo(tree.params)

    @staticmethod
    def move(command, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo("Move around locally in the tree.")
            return
        tree.index.step(command)
        click.echo(tree)

    @staticmethod
    def config(_, command_params, __):
        if command_params and command_params[0] == "help":
            click.echo("Show or modify the config 'config param=value param=value")
            return

        if command_params:
            for command in command_params:
                name, value = command.split("=")
                config.update({name: value})
        click.echo(config)

    @staticmethod
    def display(command, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo(
                "Display the tree by default, or the context, the path, the last summary, or a specific node (by calling with the nodes index)"
            )
            return

        if command == "a":
            click.echo(tree.get_full_repr())
            return

        if "context" in command_params:
            click.echo(tree.index.context)
            return

        if "path" in command_params:
            click.echo(tree.index.path)
            return

        if "prompt" in command_params:
            click.echo(tree.index.prompt)
            return

        if "summary" in command_params:
            click.echo(tree.index.last_summary)
            return

        for param in command_params:
            if param.isdigit():
                click.echo(tree.index.index_struct.all_nodes[int(command)].text)
                continue

        click.echo(tree)

    @staticmethod
    def new_node(command, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo(
                "s[end] X adds a new message to the chain and sends it all.\n"
                "\t\t\ts[end] on its own sends the existing chain as is\n"
                "\t\t\tno[send] X adds a new message to the chain but doesn't send it all"
            )
            return

        if command_params:
            prompt = command
            prompter = store.get_prompter()
            if len(tree) == 0 and prompter is not None:
                prompt = prompter.prompt(prompt)

            tree.input(prompt)
            tree.save()

        if command == "send":
            response = simple_gen(tree)
            click.echo(response)
            tree.output(response)

    @staticmethod
    def tag(_, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo(
                "tag X tags the current branch (checkout tags with checkout X)\n" "\t\t\ttag alone shows tag list"
            )
            return

        if command_params:
            "_".join(command_params)
            tree.index.tag(command_params)
        click.echo(tree.index.tags)

    @staticmethod
    def checkout(_, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo("checkout X checks out a tag")
            return

        if command_params:
            if checkout.isdigit():
                checkout = int(checkout)
            else:
                checkout = "_".join(command_params)
            tree.index.checkout(checkout)
        click.echo(tree)

    @staticmethod
    def context(_, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo(
                "modify the context. context can be added to nodes but are not part of the main path\n"
                "\t\t\tsubcommands are clear, list, remove or add"
            )
            return

        if command_params[0] == "clear":
            for node in tree.index.path:
                if node.node_info.get("context"):
                    del node.node_info["context"]

        if command_params[0] == "list":
            docs = [(tree.index.get_context(node), node.index) for node in tree.index.path]
            docs = [(doc, index) for doc, index in docs if doc is not None]
            for doc, index in docs:
                doc_text = doc.text.replace("\n", " ")[: tree.termwidth]
                click.echo(f"{index}: {doc_text}")
            return

        if command_params[0] == "remove":
            for index in command_params[1:]:
                tree.index.delete_context(int(index))

        if command_params[0] == "add":
            # If no context is given, just add the last node as context
            if len(command_params) == 1:
                new_context = tree.index.path[-1].text
            else:
                new_context = " ".join(command_params[1:])

            tree.index.add_context(new_context, tree.index.path[-1])

        tree.save()

    @staticmethod
    def edit(_, command_params, tree):
        if command_params and command_params[0] == "help":
            click.echo("Edit a node (if no node provided, edit the last one).")
            return

        if command_params and command_params[0] == "send":
            new_message = click.edit()
            Store.new_node("send", new_message.split(" "), tree)
            return

        if not command_params:
            index = tree.index.path[-1].index

        input = tree.index.index_struct.all_nodes[index].text
        output = click.edit(input)
        tree.index.index_struct.all_nodes[index].text = output
        click.echo(output)

    @staticmethod
    def help(_, command_params, ___):
        done = []
        for k, v in COMMANDS.items():
            if v == Store.help:
                continue
            if v in done:
                click.echo(f"{k}:\t\tibid.")
                continue
            click.echo(f"{k}:\t\t", nl=False)
            v("", ["help"], "")
            done.append(v)

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
            command, *command_params = command.split(" ")
            if command in COMMANDS:
                COMMANDS[command](command, command_params, tree)
            else:
                click.echo(f"Command {command} not found. Press ? or 'help' for help.")


COMMANDS = {
    "q": Store.exit,
    "p": Store.params,
    "params": Store.params,
    "h": Store.move,
    "j": Store.move,
    "k": Store.move,
    "l": Store.move,
    "d": Store.display,
    "display": Store.display,
    "a": Store.display,
    "display_all": Store.display,
    "n": lambda _, __, ___: tree.index.clear_checkout() if not len(__) else click.echo("test"),
    "new": lambda _, __, ___: tree.index.clear_checkout() if not len(__) else click.echo("test"),
    "c": Store.config,
    "config": Store.config,
    "s": Store.new_node,
    "send": Store.new_node,
    "no": Store.new_node,
    "nosend": Store.new_node,
    "nn": Store.new_node,
    "t": Store.tag,
    "tag": Store.tag,
    "context": Store.context,
    "e": Store.edit,
    "?": Store.help,
    "help": Store.help,
}


def simple_gen(tree):
    if tree.params["model"] == "trest":
        return tree.prompt
    encoding = Config.get_encoding(tree.params.get("model", "gpt-3.5-turbo"))
    tree.params["max_tokens"] -= len(encoding.encode(tree.prompt))
    if tree.params["model"] == "code-davinci-002":
        tree.params["openai_api_base"] = os.environ.get("CD2_URL")
        tree.params["openai_api_key"] = os.environ.get("CD2_KEY")
    llm = OpenAI(**tree.params)
    responses = llm.generate([tree.prompt])
    return responses.generations[0][0].text
