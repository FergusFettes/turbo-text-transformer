#!/usr/bin/env python3

from pathlib import Path

import click
from langchain.llms import OpenAI
from tttp.prompter import Prompter

from ttt.config import Config
from ttt.io import IO
from ttt.store import Store
from ttt.tree import DummyTree, Tree


def simple_gen(tree):
    llm = OpenAI(**tree.params)
    responses = llm.generate([tree.prompt])
    return responses.generations[0][0].text


def init(prompt, prompter, params):
    config = Config.check_config()
    params = Config.prepare_engine_params(params)

    if config["chat"]:
        tree = Tree(config["chat_file"], params=params)
        prompt = tree.get_and_run_commands(prompt)
    else:
        tree = DummyTree(params=params)

    # If its a simple gen or it is the first item in a new tree, apply the prompt
    if len(tree) == 0 and prompter is not None:
        prompt = prompter.prompt(prompt)

    tree.input(prompt)
    return tree


def response_length(tree):
    encoding = Config.get_encoding(tree.params.get("model", "gpt-3.5-turbo"))
    tree.params["max_tokens"] -= len(encoding.encode(tree.prompt))


@click.command()
@click.option("--toggle_chat", "-t", help="Set chat mode on/off", is_flag=True, default=False)
@click.option("--default_chat", "-d", help="Set default chat.", default="")
@click.option("--list", "-l", help="List available chats.", is_flag=True, default=False)
def store(toggle_chat, default_chat, list):
    config = Config.check_config()
    Config.check_chat(toggle_chat, default_chat, config)
    if list:
        Store(config=config).list_chats()


@click.command()
@click.option("--toggle_template", "-t", help="Set template mode on/off", is_flag=True, default=False)
@click.option("--default_template", "-d", help="Set default template.", default="")
@click.option("--list", "-l", help="List available templates.", is_flag=True, default=False)
def template(toggle_template, default_template, list):
    config = Config.check_config()
    Config.check_template(toggle_template, default_template, config)
    if list:
        Store(config=config).list_templates()


@click.command()
@click.option("--reinit", "-R", help="Recreate the config files", is_flag=True, default=False)
def config(reinit):
    Config.check_config(reinit)


@click.command()
@click.argument("prompt", required=False)
@click.option("--echo_prompt", "-e", help="Echo the pormpt in the output", is_flag=True, default=False)
@click.option("--prompt_file", "-P", help="File to load for the prompt", default=None)
@click.option("--append", "-A", help="Append to the prompt file", is_flag=True, default=False)
@click.option("--template_file", "-t", help="Template to apply to prompt.", default=None, type=str)
@click.option("--template_args", "-x", help="Extra values for the template.", default="")
@click.option("--model", "-m", help="Name of the model to use.", default="gpt-3.5-turbo")
@click.option("--number", "-N,n", help="Number of completions.", default=None, type=int)
@click.option("--max_tokens", "-M", help="Max number of tokens to return", default=None, type=int)
@click.option(
    "--temperature", "-T", help="Temperature, [0, 2]-- 0 is deterministic, >0.9 is creative.", default=None, type=int
)
def chat(
    prompt,
    echo_prompt,
    prompt_file,
    append,
    template_file,
    template_args,
    **params,
):
    prompt = IO.get_prompt(prompt, prompt_file)
    template_file = config.template_path / Path(template_file) or config.template_path / config.template_file
    prompter = Prompter.from_file(str(template_file), Config.arg2dict(template_args)) if config.template else None

    tree = init(prompt, prompter, params)
    response_length(tree)

    response = tree.prompt if params["model"] == "test" else simple_gen(tree)
    IO.return_prompt(response, prompt if echo_prompt else None, prompt_file if append else None)


@click.group()
def main():
    pass


main.add_command(chat)
main.add_command(store)
main.add_command(config)
main.add_command(template)
