#!/usr/bin/env python3

import os
from dataclasses import dataclass

from langchain.llms import OpenAI

from ttt.app import App
from ttt.app import cli as app_cli
from ttt.config import Config
from ttt.config import cli as config_cli
from ttt.config import click, shell
from ttt.store import Store
from ttt.templater import cli as templater_cli
from ttt.tree import cli as tree_cli


@click.command()
@click.argument("prompt", required=False)
@click.option("--echo_prompt", "-e", help="Echo the prompt in the output", is_flag=True, default=False)
@click.option("--prompt_file", "-P", help="File to load for the prompt", default=None)
@click.option("--append", "-A", help="Append to the prompt file", is_flag=True, default=False)
@click.option("--model", "-m", help="Name of the model to use.", default="gpt-3.5-turbo")
@click.option("--number", "-N,n", help="Number of completions.", default=None, type=int)
@click.option("--max_tokens", "-M", help="Max number of tokens to return", default=None, type=int)
@click.option(
    "--temperature", "-T", help="Temperature, [0, 2]-- 0 is deterministic, >0.9 is creative.", default=None, type=int
)
@click.pass_context
def chat(
    ctx,
    prompt,
    echo_prompt,
    prompt_file,
    append,
    **params,
):
    ctx.obj.echo_prompt = echo_prompt
    ctx.obj.append = append

    prompt = ctx.obj.io.get_prompt(prompt, prompt_file)
    prompt = ctx.obj.templater.in_(prompt)
    params = Config.prepare_engine_params(params)
    ctx.obj.tree.params.update(params)

    prompt = ctx.obj.templater.prompt(prompt)
    response = ctx.obj.simple_gen(prompt, ctx.obj.tree)
    ctx.obj.output(response)
    response = ctx.obj.templater.out(response)
    ctx.obj.tree.output(response)


@shell(prompt=">> ", intro="Starting app...")
@click.pass_context
def main(ctx):
    ctx.obj = App()


main.add_command(chat)
main.add_command(Store.file)
main.add_command(config_cli, "config")
main.add_command(config_cli, "c")
main.add_command(tree_cli, "tree")
main.add_command(tree_cli, "t")
main.add_command(app_cli, "params")
main.add_command(templater_cli, "templater")
main.add_command(templater_cli, "template")
main.add_command(templater_cli, "tr")
tree_cli.add_command(app_cli, "params")
tree_cli.add_command(templater_cli, "templater")
tree_cli.add_command(templater_cli, "template")
tree_cli.add_command(templater_cli, "tr")
