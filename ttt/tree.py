import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gpt_index import Document, GPTMultiverseIndex
from gpt_index.data_structs.data_structs import Node

from ttt.config import click, rich, shell


@dataclass
class DummyTree:
    params: Optional[dict] = None
    prompt: str = ""

    def input(self, prompt):
        self.prompt = prompt

    def output(self, *_):
        pass

    def __len__(self):
        return 0


@dataclass
class Tree:
    file: Optional[str] = None
    in_str: str = "In: "
    out_str: str = "Out: "
    index: Optional[GPTMultiverseIndex] = None
    params: Optional[dict] = None
    name: Optional[str] = None
    termwidth: int = 80

    def __post_init__(self):
        self.file = Path(self.file)
        self.termwidth = shutil.get_terminal_size().columns

        if self.file and self.file.exists():
            self.index = GPTMultiverseIndex.load_from_disk(str(self.file))
            return

        self.index = GPTMultiverseIndex(documents=[])

    @property
    def prompt(self):
        prompt = self.index.context + "\n" + self.index.__str__()
        prompt += f"\n"
        return prompt

    def input(self, prompt):
        self.extend(prompt, preface=self.in_str)

    def output(self, prompt):
        self.extend(prompt, preface=self.out_str, save=True)

    def extend(self, response, preface=None, save=False):
        if not response.startswith(preface):
            response = f"{preface}{response}"
        self.index.extend(Document(response))
        if save:
            self.save()

    def save(self):
        self.index.save_to_disk(self.file)

    def __len__(self):
        return len(self.index.index_struct.all_nodes)

    def __repr__(self) -> str:
        """Get string representation, in the manner of a git log."""
        return self._get_repr(_str=self._legend())

    def get_full_repr(self, summaries=False) -> str:
        uber_root = Node(
            index=-1,
            text="(displaying all nodes)",
            child_indices=[i for i in self.index.index_struct.root_nodes.keys()],
            node_info={},
        )
        _str = self._legend()
        _str += self._root_info()
        return self._get_repr(uber_root, _str, summaries=summaries)

    def _root_info(self) -> str:
        _str = "\n# Root Node Index (branches:total_nodes)) #\n"
        for root in self.index.index_struct.root_nodes.values():
            leaves = self.index.index_struct.get_leaves(root)
            children = self.index.index_struct.get_all_children(root)
            _str += f"{root.index}; ({len(leaves)}:{len(children)}):\t\t{root.text.splitlines()[0]}"
            _str += f"\t\t{'<-- CURRENT_ROOT' if self.index.index_struct.all_nodes[root.index].node_info.get('checked_out', False) else ''}\n"
        return _str + "\n"

    def _legend(self) -> str:
        return (
            "\n"
            "# Legend #######################################\n"
            "# * are nodes that are not checked out         #\n"
            "# X are nodes that are checked out             #\n"
            "# E are nodes that have embeddings             #\n"
            "# if you have run a similarity query,          #\n"
            "#   the similarities will be displayed         #\n"
            "#   instead of the E                           #\n"
            "# ##############################################\n"
            "\n"
            "# Conversation Tree. You can checkout indexes.##\n"
        )

    def _get_repr(self, node: Optional[Node] = None, _str: str = "", summaries: bool = False) -> str:
        if node is None:
            checked_out = [
                i for i, n in self.index.index_struct.all_nodes.items() if n.node_info.get("checked_out", False)
            ]
            if checked_out:
                node = self.index.index_struct.all_nodes[checked_out[0]]
            else:
                node = self.index.index_struct.all_nodes[min(self.index.index_struct.all_nodes.keys())]
        repr = self._get_repr_recursive(node, repr=_str, summaries=summaries)
        return repr + "\n# ##############################################\n\n"

    def _get_repr_recursive(
        self, node: Optional[Node] = None, indent: int = 0, repr: str = "", summaries: bool = False
    ) -> str:
        info_str = "*" if not node.node_info.get("checked_out", False) else "X"
        if node.embedding and node.node_info.get("similarity", None) is None:
            info_str += "E"
        if node.node_info.get("similarity", None) is not None:
            info_str += f"{node.node_info['similarity']:.2f}"

        text_width = self.termwidth - 30
        text = node.text.splitlines()[0]
        if len(text) > text_width:
            text = text[:text_width] + " ..."

        summary = ""
        if summaries and node.node_info.get("summary", None) is not None:
            summary = "\t\t"
            summary += node.node_info["summary"].replace("\n", "; ")
        info_str += f"\t\t{node.index}: {text}{summary}\n"

        many_nodes = len(self.index.index_struct.all_nodes) > 50
        space = " " if many_nodes else "  "

        prefix = indent * f"|{space}"
        repr += f"{prefix}{info_str}"

        nodes = self.index.index_struct.get_children(node)
        if not nodes:
            return repr

        if len(nodes) > 1:
            repr += f"{prefix}|\\ \n"
            if not many_nodes:
                repr += f'{indent * "|  "}| \\ \n'

        for child_node in nodes.values():
            if self.index.index_struct.is_last_child(child_node):
                repr = self._get_repr_recursive(child_node, indent, repr, summaries)
            else:
                repr = self._get_repr_recursive(child_node, indent + 1, repr, summaries)

        return repr


@shell(prompt="tree> ")
@click.pass_context
def cli(ctx):
    """Manage app config."""
    rich.print(ctx.obj.tree)


@cli.command()
@click.argument("count", default=1)
@click.pass_context
def h(ctx, count):
    "Move to left sibling"
    for _ in range(count):
        ctx.obj.tree.index.step("h")
    rich.print(ctx.obj.tree)


@cli.command()
@click.argument("count", default=1)
@click.pass_context
def j(ctx, count):
    "Move to parent"
    for _ in range(count):
        ctx.obj.tree.index.step("j")
    rich.print(ctx.obj.tree)


@cli.command()
@click.argument("count", default=1)
@click.pass_context
def k(ctx, count):
    "Move to child"
    for _ in range(count):
        ctx.obj.tree.index.step("k")
    rich.print(ctx.obj.tree)


@cli.command()
@click.argument("count", default=1)
@click.pass_context
def l(ctx, count):
    "Move to right sibling"
    for _ in range(count):
        ctx.obj.tree.index.step("l")
    rich.print(ctx.obj.tree)


@cli.command()
@click.argument("type", default="tree")
@click.argument("index", default=None, required=False)
@click.pass_context
def display(ctx, type, index):
    """Display the tree.\n
    Types:\n
        \t[default] tree: display the tree structure\n
        \tall/a: display the full tree including other roots\n
        \tpath/p: display the path to the current node\n
        \tsummary/s: display the current context and latest summary\n
        \tnode/n: display the specific node(s) (pass the index of the node(s))
    """
    if type in ["t", "tree"]:
        rich.print(ctx.obj.tree)

    if type in ["a", "all"]:
        rich.print(ctx.obj.tree.get_full_repr())
        return

    if type in ["c", "context"]:
        rich.print(ctx.obj.tree.index.context)
        return

    if type in ["p", "path"]:
        rich.print(ctx.obj.tree.index.path)
        return

    if type in ["s", "summary"]:
        rich.print(ctx.obj.tree.index.last_summary)
        return

    if type in ["n", "node"]:
        if index is None:
            rich.print("Please provide an index")
            return

        if "," in index:
            indexes = index.split(",")
        else:
            indexes = [index]

        for index in indexes:
            if index.isdigit():
                rich.print(ctx.obj.tree.index.index_struct.all_nodes[int(index)].text)
                continue


cli.add_command(display, "d")


@cli.command()
@click.argument("msg", default=None, required=False)
@click.pass_context
def send(ctx, msg):
    """s[end] MSG adds a new message to the chain and sends it all.\n
    s[end] on its own sends the existing chain as is"""

    if msg:
        _append(ctx, msg)

    response = ctx.obj.simple_gen(ctx.obj.tree)
    rich.print(response)
    ctx.obj.tree.output(response)


cli.add_command(send, "s")


@cli.command()
@click.argument("msg", default=None, required=False)
@click.pass_context
def append(ctx, msg):
    """ap[pend] MSG adds a new node at the end of the chain. If MSG is empty, an editor will be opened."""
    if not msg:
        msg = click.edit()
    _append(ctx, msg)


cli.add_command(append, "ap")


def _append(ctx, msg):
    prompter = ctx.obj.store.get_prompter()
    if len(ctx.obj.tree) == 0 and prompter is not None:
        msg = prompter.prompt(msg)

    ctx.obj.tree.input(msg)
    ctx.obj.tree.save()


@cli.command()
@click.argument("tag", default=None, required=False)
@click.pass_context
def tag(ctx, tag):
    "tag X tags the current branch (checkout tags with checkout X)\n\t\ttag alone shows tag list"
    if tag:
        ctx.obj.tree.index.tag(tag)
    rich.print(ctx.obj.tree.index.tags)


cli.add_command(tag, "t")


@cli.command()
@click.argument("tag", default=None)
@click.pass_context
def checkout(ctx, tag):
    "checkout X checks out a tag or index"
    if tag.isdigit():
        tag = int(tag)
    ctx.obj.tree.index.checkout(tag)
    rich.print(ctx.obj.tree)


cli.add_command(checkout, "c")


@cli.command()
@click.argument("index", default=None, required=False)
@click.pass_context
def edit(ctx, index):
    "Edit a node (if no node provided, edit the last one)."

    if not index:
        index = ctx.obj.tree.index.path[-1].index
    index = int(index)

    input = ctx.obj.tree.index.index_struct.all_nodes[index].text
    output = click.edit(input)
    ctx.obj.tree.index.index_struct.all_nodes[index].text = output
    rich.print(output)


cli.add_command(edit, "e")


@cli.command()
@click.argument("indexes", default=None, required=False)
@click.pass_context
def delete(ctx, indexes):
    "Delete a node (if no node provided, delete the last one)."

    if not indexes:
        indexes = [ctx.obj.tree.index.path[-1].index]
    else:
        indexes = indexes.split(",")

    for index in indexes:
        ctx.obj.tree.index.delete(int(index))


cli.add_command(delete, "del")


@cli.command()
@click.argument("indexes", default=None)
@click.pass_context
def cherry_pick(ctx, indexes):
    "Copy nodes onto the current branch (can be indexes or tags)"
    indexes = [int(index) if index.isdigit() else index for index in indexes.split(",")]
    ctx.obj.tree.index.cherry_pick(indexes)


cli.add_command(cherry_pick, "cp")


# @staticmethod
# def context(_, command_params, tree):
#     if command_params and command_params[0] == "help":
#         click.echo(
#             "modify the context. context can be added to nodes but are not part of the main path\n"
#             "\t\t\tsubcommands are clear, list, remove or add"
#         )
#         return
#
#     if command_params[0] == "clear":
#         for node in tree.index.path:
#             if node.node_info.get("context"):
#                 del node.node_info["context"]
#
#     if command_params[0] == "list":
#         docs = [(tree.index.get_context(node), node.index) for node in tree.index.path]
#         docs = [(doc, index) for doc, index in docs if doc is not None]
#         for doc, index in docs:
#             doc_text = doc.text.replace("\n", " ")[: tree.termwidth]
#             click.echo(f"{index}: {doc_text}")
#         return
#
#     if command_params[0] == "remove":
#         for index in command_params[1:]:
#             tree.index.delete_context(int(index))
#
#     if command_params[0] == "add":
#         # If no context is given, just add the last node as context
#         if len(command_params) == 1:
#             new_context = tree.index.path[-1].text
#         else:
#             new_context = " ".join(command_params[1:])
#
#         tree.index.add_context(new_context, tree.index.path[-1])
#
#     tree.save()
