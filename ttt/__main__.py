#!/usr/bin/env python3

from pathlib import Path

import click
from dotenv import load_dotenv
from gpt_index import GPTListIndex
from gpt_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

from ttt.chunker import Chunker
from ttt.config import arg2dict, config, config_path, create_config, load_config, save_config
from ttt.models import BaseModel, OpenAIModel

load_dotenv()


def run_chat(prompt, model, file):
    llm = OpenAI(model=model)
    if file and Path(file).exists():
        index = GPTListIndex.load_from_disk(file)
    else:
        index = GPTListIndex([], text_splitter=CharacterTextSplitter())
    memory = GPTIndexChatMemory(
        index=index,
        memory_key="chat_history",
        query_kwargs={"response_mode": "compact"},
        # return_source returns source nodes instead of querying index
        return_source=True,
    )
    agent_chain = initialize_agent([], llm, agent="conversational-react-description", memory=memory)
    response = agent_chain.run(input=prompt)
    index.save_to_disk(file)
    return "\n" + response


def check_config(reinit):
    """Check that the config file exists."""
    if not reinit and config_path.exists():
        return config

    click.echo("Config file not found. Creating one for you...", err=True, color="red")
    create_config()
    openai_api_key = click.prompt("OpenAI API Key", type=str)
    if openai_api_key:
        OpenAIModel.create_config(openai_api_key)

    # Same for other providers...

    return load_config()


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


def prepare_engine_params(params, format):
    """Prepare options for the OpenAI API."""
    params = {k: v for k, v in params.items() if v is not None}

    params["token_limit"] = 4000 if params["model"] in OpenAIModel().large_models else 2048
    if "number" in params:
        params["n"] = params.pop("number")
    if "logprobs" in params and params["logprobs"] == 0:
        if format == "logprobs":
            # If format is logprobs, we need to get the logprobs
            params["logprobs"] = 1
        else:
            params["logprobs"] = None
    return params


def get_prompt(prompt, prompt_file, params):
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
        params["force"] = True
    return prompt


def chunk(prompt, verbose, params):
    prompt_args = arg2dict(params["template_args"])
    chunker = Chunker(prompt, verbose=verbose, params=params)
    if chunker.needs_chunking():
        if not params["force"]:
            click.confirm("Do you want to chunk the prompt?", abort=True, err=True)
        click.echo("Chunking...", err=True)
        return chunker.chunk()
    if chunker.template_size:
        return [chunker.prompter.prompt(prompt, prompt_args)]
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
@click.option("--toggle_chat", "-H", help="Set chat mode on/off", is_flag=True, default=False)
@click.option("--default_chat", "-d", help="Set default chat.", default="")
@click.option("--reinit", "-R", help="Recreate the config files", is_flag=True, default=False)
@click.option("--echo_prompt", "-e", help="Echo the pormpt in the output", is_flag=True, default=False)
@click.option("--cost_only", "-C", help="Estimate the cost of the query", is_flag=True, default=False)
@click.option("--verbose", "-v", help="Verbose output", is_flag=True, default=False)
@click.option("--list_models", "-l", help="List available models.", is_flag=True, default=False)
@click.option("--prompt_file", "-P", help="File to load for the prompt", default=None)
@click.option("--append", "-A", help="Append to the prompt file", is_flag=True, default=False)
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
@click.option("--force", "-F", help="Force chunking of prompt", is_flag=True, default=False)
def main(
    prompt,
    format,
    toggle_chat,
    default_chat,
    reinit,
    echo_prompt,
    cost_only,
    verbose,
    list_models,
    prompt_file,
    append,
    **params,
):
    # click.echo(params, err=True)
    config = check_config(reinit)
    check_chat(toggle_chat, default_chat, config)
    params = prepare_engine_params(params, format)
    options = {"params": params, "format": format, "echo_prompt": echo_prompt}

    sink = click.get_text_stream("stdout")
    oam = OpenAIModel(**options)
    if list_models:
        sink.write("\n".join(oam.list))
        return

    prompt = get_prompt(prompt, prompt_file, params)

    if config["chat"]:
        response = run_chat(prompt, params["model"], config["chat_file"])
        sink.write(response)
        return

    if cost_only:
        verbose = True
    prompts = chunk(prompt, verbose, params)
    if cost_only:
        return

    if params["model"] in oam.list:
        responses = [oam.gen(prompt) for prompt in prompts]
        if append:
            with open(prompt_file, "a") as f:
                f.write("\n".join(responses))
            return
        sink.write("\n".join(responses))
        return

    if params["model"] == "test":
        bm = BaseModel(**options)
        responses = [bm.gen(prompt) for prompt in prompts]
        sink.write("\n".join(responses))
        return

    click.echo("Model not found. Use the -l flag to list available models.", err=True, color="red")
