from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jinja2
from rich.panel import Panel
from rich.table import Table

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

    def in_(self, message):
        in_prefix = self.template_config["in_prefix"] or ""
        in_postfix = self.template_config["in_postfix"] or ""
        return in_prefix + message + in_postfix

    def prompt(self, prompt):
        out_prefix = self.template_config["out_prefix"] or ""
        prompt = prompt + out_prefix
        if self.template_config["template"]:
            prompt = self._prompt(prompt)
        return prompt

    def _prompt(self, prompt):
        args = {"prompt": prompt}
        template = Path(self.template_file).read_text()
        return jinja2.Template(template).render(**args)

    def out(self, message):
        out_prefix = self.template_config["out_prefix"] or ""
        out_postfix = self.template_config["out_postfix"] or ""
        return out_prefix + message + out_postfix

    def save(self):
        config = self.config
        config["templater"] = self.template_config
        Config.save_config(self.config)

    def list_templates(self, short):
        if not self.template_path.exists():
            return

        files = [x for x in self.template_path.glob("*.j2")]
        click.echo(f"Found {len(files)} templates.")
        if short:
            table = Table("Filename", "Text", box=rich.box.MINIMAL_DOUBLE_HEAD, show_lines=True)

        for file in files:
            if short:
                table.add_row(file.stem, file.read_text().replace("\n", "\\n"))
            else:
                rich.print(Panel(file.read_text(), title=file.stem, border_style="blue"))

        if short:
            rich.print(table)


@shell(prompt="templater> ")
@click.pass_context
def cli(ctx):
    if Path(ctx.obj.templater.template_file).exists():
        contents = Path(ctx.obj.templater.template_file).read_text()
        rich.print(Panel(contents, title=ctx.obj.templater.template_file, border_style="blue"))
    else:
        rich.print(f"Template file {ctx.obj.templater.template_file} does not exist.")


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
@click.argument("filename")
@click.pass_context
def default(ctx, filename):
    if not filename.endswith(".j2"):
        filename = filename + ".j2"
    ctx.obj.templater.template_config["template_file"] = filename
    ctx.obj.templater.save()


cli.add_command(default, "d")


@cli.command()
@click.argument("prefix", default="", required=False)
@click.argument("postfix", default="", required=False)
@click.pass_context
def in_(ctx, prefix, postfix):
    if not postfix:
        postfix = "\n" if prefix else ""
    ctx.obj.templater.template_config["in_prefix"] = prefix
    ctx.obj.templater.template_config["in_postfix"] = postfix
    ctx.obj.templater.save()


cli.add_command(in_, "in")


@cli.command()
@click.argument("prefix", default="", required=False)
@click.argument("postfix", default="", required=False)
@click.pass_context
def out(ctx, prefix, postfix):
    if not postfix:
        postfix = "\n" if prefix else ""
    ctx.obj.templater.template_config["out_prefix"] = prefix
    ctx.obj.templater.template_config["out_postfix"] = postfix
    ctx.obj.templater.save()


@cli.command()
@click.option("--short", "-s", default=False, is_flag=True)
@click.pass_context
def list(ctx, short):
    ctx.obj.templater.list_templates(short)


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
    click.edit(filename=filename)

    if filename.stem != Path(ctx.obj.templater.template_file).stem:
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


@cli.command()
@click.argument("filename", default=None)
@click.pass_context
def show(ctx, filename):
    """Show a template."""
    if filename is None:
        filename = Path(ctx.obj.templater.template_file).stem
    if not filename.endswith(".j2"):
        filename = filename + ".j2"
    filename = ctx.obj.templater.template_path / Path(filename)

    rich.print(Panel.fit(filename.read_text(), title=filename.stem, border_style="blue"))
