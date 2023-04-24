import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gpt_index import Document, GPTMultiverseIndex
from gpt_index.data_structs.data_structs import Node


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
