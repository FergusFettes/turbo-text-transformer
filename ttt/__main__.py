#!/usr/bin/env python3

import click
from langchain.llms import OpenAI

from ttt.config import Config
from ttt.io import IO
from ttt.store import Store


def simple_gen(tree):
    response_length(tree)
    llm = OpenAI(**tree.params)
    responses = llm.generate([tree.prompt])
    return responses.generations[0][0].text


def response_length(tree):
    encoding = Config.get_encoding(tree.params.get("model", "gpt-3.5-turbo"))
    tree.params["max_tokens"] -= len(encoding.encode(tree.prompt))


@click.command()
@click.option("--reinit", "-R", help="Recreate the config files", is_flag=True, default=False)
def config(reinit):
    Config.check_config(reinit)


@click.command()
@click.argument("prompt", required=False)
@click.option("--echo_prompt", "-e", help="Echo the prompt in the output", is_flag=True, default=False)
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
    config = Config.check_config()
    store = Store(config=config)
    prompter = store.get_prompter(template_file, template_args)
    tree = store.get_tree(prompt, prompter, params)

    response = tree.prompt if params["model"] == "test" else simple_gen(tree)
    IO.return_prompt(response, prompt if echo_prompt else None, prompt_file if append else None)
    tree.output(response)


@click.group()
def main():
    pass


main.add_command(chat)
main.add_command(Store.file)
main.add_command(config)
main.add_command(Store.template)
main.add_command(Store.tree)
