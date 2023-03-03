#!/usr/bin/env python3

import click

from ttt.models import BaseModel, OpenAIModel


@click.command()
@click.option("--model", "-m", help="Name of the model to use.", default="gpt-3.5-turbo-0301")
@click.option("--prompt", "-p", help="Prompt to use.", default="")
@click.option("--number", "-n", help="Number of completions.", default=1)
@click.option("--list_models", "-l", help="List available models.", is_flag=True)
@click.option(
    "--format",
    "-f",
    help="Output Format",
    default="clean",
    type=click.Choice(["clean", "json", "logprobs"]),
    show_default=True,
)
def main(model, prompt, number, list_models, format):
    # If there is no prompt, try to get it from stdin
    if not prompt:
        prompt = click.get_text_stream("stdin").read().strip()
        if not prompt:
            return

    sink = click.get_text_stream("stdout")

    oam = OpenAIModel(model=model, params={"n": number}, format=format)
    if list_models:
        sink.write("\n".join(oam.list))
        return

    bm = BaseModel(model="test", params={"n": number}, format=format)
    if model in oam.list:
        completion = oam.gen(prompt)
    else:
        completion = bm.gen(prompt)

    sink.write(completion)
