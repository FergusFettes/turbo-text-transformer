#!/usr/bin/env python3

import click

from ttt.models import BaseModel, OpenAIModel


@click.command()
@click.option("--model", "-m", help="Name of the model to use.", default="davinci")
@click.option("--prompt", "-p", help="Prompt to use.", default="")
@click.option("--number", "-n", help="Number of completions.", default=1)
def main(model, prompt, number):
    # If there is no prompt, try to get it from stdin
    if not prompt:
        prompt = click.get_text_stream("stdin").read().strip()
        if not prompt:
            return

    sink = click.get_text_stream("stdout")

    oam = OpenAIModel(model=model, params={"n": number})
    bm = BaseModel(model="test", params={"n": number})
    if model in oam.list:
        completion = oam.gen(prompt)
    else:
        completion = bm.gen(prompt)

    sink.write("\n".join(completion))
