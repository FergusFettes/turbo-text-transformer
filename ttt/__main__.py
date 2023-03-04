#!/usr/bin/env python3

import click
from ttt.chunker import Chunker

from ttt.config import config_path, create_config, load_config, encoding
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


def chunk(prompt, params):
    prompt_tokens = encoding.encode(prompt)
    max_tokens = 4000 if params["model"] in OpenAIModel.large_models else 2048
    if len(prompt_tokens) > max_tokens:
        click.echo(f"Prompt is too long. Max tokens: {max_tokens}. Prompt tokens: {len(prompt_tokens)}")
        click.confirm("Do you want to chunk the prompt?", abort=True)
        return Chunker(prompt, **params).chunk()

    if len(prompt_tokens) + int(params["max_tokens"]) > max_tokens:
        params["max_tokens"] = max_tokens - len(prompt_tokens)
        click.echo(f"Prompt is too long. Max tokens adjusted: {max_tokens}. Prompt tokens: {len(prompt_tokens)}")
    return [prompt]


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
@click.option("--echo_prompt", "-e", help="List available models.", is_flag=True, default=False)
@click.option("--list_models", "-l", help="List available models.", is_flag=True, default=False)
@click.option("--template_file", "-t", help="Template to apply to prompt.", default=None, type=str)
@click.option("--chunk_size", "-c", help="Max size of chunks", default=None, type=int)
@click.option("--summary_size", "-s", help="Size of chunk summaries", default=None, type=int)
@click.option("--model", "-M", help="Name of the model to use.", default="gpt-3.5-turbo")
@click.option("--number", "-N", help="Number of completions.", default=None, type=int)
@click.option("--logprobs", "-L", help="Show logprobs for completion", default=None, type=int)
@click.option("--max_tokens", "-M", help="Max number of tokens to return", default=None, type=int)
@click.option("--temperature", "-T", help="Temperature, [0, 2]-- 0 is deterministic, >0.9 is creative.", default=None, type=int)
def main(prompt, format, echo_prompt, list_models, **params):
    check_config()
    options = {"params": prepare_engine_params(params, format), "format": format, "echo_prompt": echo_prompt}

    sink = click.get_text_stream("stdout")
    oam = OpenAIModel(**options)
    if list_models:
        sink.write("\n".join(oam.list))
        return

    prompt = get_prompt(prompt)
    prompts = chunk(prompt)
    if params["model"] in oam.list:
        responses = [oam.gen(prompt) for prompt in prompts]
        sink.write('\n'.join(responses))
        return

    if params["model"] == 'test':
        bm = BaseModel(**options)
        responses = [bm.gen(prompt) for prompt in prompts]
        sink.write('\n'.join(responses))

    click.echo("Model not found. Use the -l flag to list available models.", err=True, color="red")
