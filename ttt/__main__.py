#!/usr/bin/env python3

import click

from ttt.config import config_path, create_config, load_config
from ttt.models import BaseModel, OpenAIModel


def check_config():
    """Check that the config file exists."""
    if config_path.exists():
        return

    click.echo("Config file not found. Creating one for you...", err=True, color="red")
    create_config()
    openai_api_key = click.prompt("OpenAI API Key", type=str)
    if openai_api_key:
        OpenAIModel.create_config(openai_api_key)

    # Same for other providers...

    load_config()


def prepare_engine_params(params, format):
    """Prepare options for the OpenAI API."""
    params = {k: v for k, v in params.items() if v is not None}
    if "number" in params:
        params["n"] = params.pop("number")
    if "logprobs" in params and params["logprobs"] == 0:
        if format == "logprobs":
            # If format is logprobs, we need to get the logprobs
            params["logprobs"] = 1
        else:
            params["logprobs"] = None
    return params


def get_prompt(prompt):
    """Get the prompt from stdin if it's not provided."""
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
@click.option("--model", "-m", help="Name of the model to use.", default="gpt-3.5-turbo-0301")
@click.option(
    "--format",
    "-f",
    help="Output Format",
    default="clean",
    type=click.Choice(["clean", "json", "logprobs"]),
    show_default=True,
)
@click.option("--list_models", "-l", help="List available models.", is_flag=True, default=False)
@click.option("--echo_prompt", "-e", help="List available models.", is_flag=True, default=False)
@click.option("--number", "-n", help="Number of completions.", default=None, type=int)
@click.option("--logprobs", "-L", help="Show logprobs for completion", default=None, type=int)
@click.option("--max_tokens", "-M", help="Max number of tokens to return", default=None, type=int)
def main(prompt, model, format, echo_prompt, list_models, **params):
    check_config()
    options = {"params": prepare_engine_params(params, format), "format": format, "echo_prompt": echo_prompt}

    oam = OpenAIModel(model=model, **options)
    if list_models:
        click.get_text_stream("stdout").write("\n".join(oam.list))
        return

    prompt = get_prompt(prompt)
    sink = click.get_text_stream("stdout")
    if model in oam.list:
        sink.write(oam.gen(prompt))
        return

    bm = BaseModel(model="test", **options)
    sink.write(bm.gen(prompt))
