from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from tttp.prompter import Prompter

from ttt.config import Config, click, rich, shell


@dataclass
class Templater:
    """
    Templates are stored as editable files.
    """

    config: Optional["Config"] = None  # type: ignore
    template_config: Optional["Config"] = None  # type: ignore

    def __post_init__(self):
        if self.config:
            self.template_config = self.config["templater"]

    @property
    def template_file(self):
        """The template_file property."""
        return str(self.template_path / self.template_config["template_file"])

    @template_file.setter
    def template_file(self, value):
        self.template_config["template_file"] = str(value)

    @property
    def template_path(self):
        """The template_path property."""
        return Path(self.template_config["template_path"])

    @template_path.setter
    def template_path(self, value):
        self.template_config["template_file"] = str(value)

    def get_prompter(self, template_file=None, template_args=None):
        if not self.template_config["template"]:
            return None
        if template_file:
            template_file = self.template_path / Path(template_file)
        else:
            template_file = self.template_file
        if template_file.exists():
            return Prompter(template_file, args=Config.arg2dict(template_args))

    def in_(self, message):
        in_prefix = self.template_config["in_prefix"] or ""
        return in_prefix + message

    def prompt(self, prompt):
        if self.template_config["out_prefix"]:
            prompt += "\n"
            prompt = prompt + self.template_config["out_prefix"]
        if self.template_config["template"]:
            prompt = Prompter(self.template_file).prompt(prompt)
        return prompt

    def out(self, message):
        out_prefix = self.template_config["out_prefix"] or ""
        return out_prefix + message

    def list_templates(self):
        if not self.template_path.exists():
            return

        files = [x for x in self.template_path.glob("*.j2")]
        click.echo(f"Found {len(files)} templates.")
        for file in files:
            rich.print(Panel(file.read_text(), title=file.stem, border_style="blue"))


@shell(prompt="templater> ")
def cli():
    pass


@cli.command()
@click.pass_context
def print(ctx):
    "Print the current config."
    rich.print(ctx.obj.templater.template_config)


cli.add_command(print, "p")


@cli.command()
@click.pass_context
def save(ctx):
    "Save the current config to the config file."
    config = ctx.obj.templater.config
    config["templater"] = ctx.obj.templater.template_config
    Config.save_config(ctx.obj.templater.config)


cli.add_command(save, "s")


@cli.command()
@click.argument("name", required=False)
@click.argument("value", required=False)
@click.argument("kv", required=False)
@click.pass_context
def update(ctx, name, value, kv):
    "Update a config value, or set of values. (kv in the form of 'name1=value1,name2=value2')"
    if kv:
        updates = kv.split(",")
        for kv in updates:
            name, value = kv.split("=")
            _update(name, value, ctx.obj.templater.template_config)
        return
    _update(name, value, ctx.obj.templater.template_config)


cli.add_command(update, "u")


def _update(key, value, dict):
    if value in ["True", "False", "true", "false"]:
        value = value in ["True", "true"]
    elif value is None:
        value = None
    elif value in ["None"]:
        value = None
    elif value.isdigit():
        value = int(value)
    elif value.replace(".", "").isdigit():
        value = float(value)

    if key == "template_file":
        if not value.endswith(".j2"):
            value = value + ".j2"
    dict.update({key: value})


@cli.command()
@click.pass_context
def toggle(ctx):
    ctx.obj.templater.template_config["template"] = not ctx.obj.templater.template_config["template"]
    click.echo(f"Template mode is {'on' if ctx.obj.templater.template_config['template'] else 'off'}.")
    config = ctx.obj.templater.config
    config["templater"] = ctx.obj.templater.template_config
    Config.save_config(ctx.obj.templater.config)


@cli.command()
@click.argument("default")
@click.pass_context
def default(ctx, default):
    if not filename.endswith(".j2"):
        filename = filename + ".j2"
    ctx.obj.templater.template_config["template_file"] = default
    config = ctx.obj.templater.config
    config["templater"] = ctx.obj.templater.template_config
    Config.save_config(ctx.obj.templater.config)


@cli.command()
@click.pass_context
def list(ctx):
    ctx.obj.templater.list_templates()


cli.add_command(list, "l")
cli.add_command(list, "ls")


@cli.command()
@click.argument("filename", default=None, required=False)
@click.pass_context
def edit(ctx, filename):
    """Edit a template file."""
    if filename is None:
        filename = Path(ctx.obj.templater.template_file).stem
    if not filename.endswith(".j2"):
        filename = filename + ".j2"
    filename = ctx.obj.templater.template_path / Path(filename)
    if filename.exists():
        click.edit(filename=filename)

        default = click.confirm(f"Make default?", abort=True)
        if default:
            ctx.obj.config["template_file"] = filename.stem
            Config.save_config(ctx.obj.config)


cli.add_command(edit, "e")


@cli.command()
@click.argument("filename", default=None)
@click.pass_context
def new(ctx, filename):
    """New template."""
    if not filename.endswith(".j2"):
        filename = filename + ".j2"
    filename = ctx.obj.templater.template_path / Path(filename)

    click.edit(filename=filename)

    default = click.confirm(f"Make default?", abort=True)
    if default:
        ctx.obj.config["template_file"] = filename.stem
        Config.save_config(ctx.obj.config)


cli.add_command(new, "n")
