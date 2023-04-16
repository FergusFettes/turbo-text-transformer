#!/usr/bin/env python3

from pathlib import Path

import click
from dotenv import load_dotenv
from langchain.llms import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
from tttp.prompter import Prompter

# from ttt.chunker import Chunker
from ttt.config import arg2dict, check_config, get_encoding, model_tokens, save_config
from ttt.tree import DummyTree, Tree

load_dotenv()


def check_chat(toggle, default, config):
    if toggle:
        config["chat"] = not config["chat"]
        click.echo(f"Chat mode is now {'on' if config['chat'] else 'off'}.", err=True)

    if default:
        config["chat_file"] = default

    if config["chat"] or config["chat_file"]:
        save_config(config)

    if config["chat"]:
        click.echo(f"Chat mode is on. Using {config['chat_file']} as the chat history file.", err=True)


def prepare_engine_params(params):
    """Prepare options for the OpenAI API."""
    params = {k: v for k, v in params.items() if v is not None}

    params["max_tokens"] = model_tokens.get(params["model"], 2048)
    if "number" in params:
        params["n"] = params.pop("number")
    return params


def get_prompt(prompt, prompt_file):
    """Get the prompt from stdin if it's not provided."""
    if prompt_file:
        prompt = Path(prompt_file).read_text().strip()
        return prompt
    if not prompt:
        click.echo("Reading from stdin... (Ctrl-D to end)", err=True)
        prompt = click.get_text_stream("stdin").read().strip()
        click.echo("Generating...", err=True)
        if not prompt:
            click.echo("No prompt provided. Use the -p flag or pipe a prompt to stdin.", err=True, color="red")
            raise click.Abort()
    return prompt


def simple_gen(tree):
    llm = OpenAI(**tree.params)
    responses = llm.generate([tree.prompt])
    return responses.generations[0][0].text


def init(reinit, toggle_chat, default_chat, prompt, prompter, params):
    # click.echo(params, err=True)
    config = check_config(reinit)
    check_chat(toggle_chat, default_chat, config)
    params = prepare_engine_params(params)

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


def handle_return(response, prompt, prompt_file):
    if prompt_file:
        with open(prompt_file, "a") as f:
            f.write("\n".join(response))
        return
    sink = click.get_text_stream("stdout")
    if prompt:
        sink.write(prompt + "\n")
    sink.write(response)


def response_length(tree):
    encoding = get_encoding(tree.params.get("model", "gpt-3.5-turbo"))
    tree.params["max_tokens"] -= len(encoding.encode(tree.prompt))


@click.command()
@click.argument("prompt", required=False)
@click.option("--toggle_chat", "-H", help="Set chat mode on/off", is_flag=True, default=False)
@click.option("--default_chat", "-d", help="Set default chat.", default="")
@click.option("--reinit", "-R", help="Recreate the config files", is_flag=True, default=False)
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
def main(
    prompt,
    toggle_chat,
    default_chat,
    reinit,
    echo_prompt,
    prompt_file,
    append,
    template_file,
    template_args,
    **params,
):
    prompt = get_prompt(prompt, prompt_file)
    prompter = Prompter.from_file(template_file, arg2dict(template_args)) if template_file else None
    tree = init(reinit, toggle_chat, default_chat, prompt, prompter, params)
    response_length(tree)

    response = tree.prompt if params["model"] == "test" else simple_gen(tree)
    handle_return(response, prompt if echo_prompt else None, prompt_file if append else None)
