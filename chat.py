#!/usr/bin/env python

from pathlib import Path

import click
from dotenv import load_dotenv
from gpt_index import GPTMultiverseIndex
from gpt_index.readers.schema.base import Document
from langchain.llms import OpenAI
from tttp.prompter import Prompter

from ttt.config import arg2dict, config, config_path, create_config, load_config, save_config

load_dotenv()


def arg2dict(args):
    d = {}
    if "=" not in args:
        return d
    for arg in args.split(","):
        k, v = arg.split("=")
        d[k] = v
    return d


def chat_history(prompt, file, params):
    if file and Path(file).exists():
        index = GPTMultiverseIndex.load_from_disk(file)
        index.extend(Document(f"In: {prompt}"))
    else:
        if params["template_file"]:
            file = Prompter.find_file(params["template_file"])
            prompter = Prompter(file)
            prompt = prompter.prompt(prompt, arg2dict(params["template_args"]))
        index = GPTMultiverseIndex([Document(prompt)])
    return index


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


def check_config(reinit):
    """Check that the config file exists."""
    if not reinit and config_path.exists():
        return config


model_tokens = {
    "gpt-4": 4096,
    "gpt-3.5-turbo": 4096,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
    "code-davinci-002": 4096,
}


def prepare_engine_params(params, format):
    """Prepare options for the OpenAI API."""
    params = {k: v for k, v in params.items() if v is not None}

    params["max_tokens"] = model_tokens.get(params["model"], 2048)
    if "number" in params:
        params["n"] = params.pop("number")
    if "logprobs" in params and params["logprobs"] == 0:
        if format == "logprobs":
            # If format is logprobs, we need to get the logprobs
            params["logprobs"] = 1
        else:
            params["logprobs"] = None
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


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--format",
    "-f",
    help="Output Format",
    default="clean",
    type=click.Choice(["clean", "json", "logprobs"]),
    show_default=True,
)
@click.option("--toggle_chat", "-H", help="Set chat mode on/off", is_flag=True, default=False)
@click.option("--default_chat", "-d", help="Set default chat.", default="")
@click.option("--reinit", "-R", help="Recreate the config files", is_flag=True, default=False)
@click.option("--prompt_file", "-P", help="File to load for the prompt", default=None)
@click.option("--template_file", "-t", help="Template to apply to prompt.", default=None, type=str)
@click.option("--template_args", "-x", help="Extra values for the template.", default="")
@click.option("--chunk_size", "-c", help="Max size of chunks", default=None, type=int)
@click.option("--summary_size", "-s", help="Size of chunk summaries", default=None, type=int)
@click.option("--model", "-m", help="Name of the model to use.", default="gpt-3.5-turbo")
@click.option("--number", "-N,n", help="Number of completions.", default=None, type=int)
@click.option("--logprobs", "-L", help="Show logprobs for completion", default=None, type=int)
@click.option("--max_tokens", "-M", help="Max number of tokens to return", default=None, type=int)
@click.option(
    "--temperature", "-T", help="Temperature, [0, 2]-- 0 is deterministic, >0.9 is creative.", default=None, type=int
)
def main(
    prompt,
    format,
    toggle_chat,
    default_chat,
    reinit,
    prompt_file,
    template_file,
    template_args,
    **params,
):
    click.echo(params, err=True)
    config = check_config(reinit)
    check_chat(toggle_chat, default_chat, config)
    params = prepare_engine_params(params, format)

    sink = click.get_text_stream("stdout")

    prompt = get_prompt(prompt, prompt_file)

    if config["chat"]:
        history = chat_history(
            prompt, config["chat_file"], {"template_file": template_file, "template_args": template_args}
        )
        prompt = history.__str__()

    params["max_tokens"] = 2048
    llm = OpenAI(**params)
    responses = llm.generate([prompt])
    response = responses.generations[0][0].text

    if response.startswith("Out: "):
        response = response[5:]

    sink.write(response)

    if config["chat"]:
        history.extend(Document(f"Out: {response}"))
        history.save_to_disk(config["chat_file"])


if __name__ == "__main__":
    main()
