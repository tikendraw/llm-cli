import os
import sys
from dataclasses import asdict, fields
from typing import Dict, List, Optional

import click
from rich.console import Console

from cli.db_utils import (
    delete_chat_session,
    get_chat_history,
    init_db,
    save_chat_history,
)
from cli.utils import (
    filter_command,
    get_context_from_file,
    get_message_or_stdin,
    handle_image_command,
    prepare_messages,
    reset_conversation,
    show_messages,
)
from core.chat import chat as cc
from core.chat import parse_image
from core.config import ChatConfig, load_config, save_config
from core.prompt import get_formatter, get_prompt

console = Console()
configg = load_config()

init_db()


@click.group
def cli():
    """ llm-cli application """
    pass


@cli.command()
@click.argument('message', type=str, required=False)
@click.option('--system_prompt', '-s', default=None, help='System prompt for the LLM.')
@click.option('--model', '-m', default=configg.model, help='Model name, e.g., provider/model_name.')
@click.option('--temperature', '-t', default=configg.temperature, help='LLM temperature (0-1).')
@click.option('--image', type=click.Path(exists=True), default=None, help='Image file path or URL.')
@click.option('--context', '-c', type=str, default=None, help='Context string or file path.')
@click.option('--no_system_prompt', is_flag=True, help='Disable system prompt.')
@click.option('--skip_vision_check', is_flag=True, help='Skip vision model check.')
def chat(message, system_prompt, model, temperature, image, context, no_system_prompt, skip_vision_check):
    """CLI-based chat interaction."""
    message = get_message_or_stdin(message)
    if not message:
        console.print("[red]No message provided. Provide a message or pipe input.[/red]")
        return

    if context:
        context = get_context_from_file(context) or context
        message = get_formatter('rag').format(question=message, context=context)
        system_prompt = get_prompt('rag') if not no_system_prompt else None

    try:
        messages = prepare_messages(message, system_prompt, image, model, context, no_system_prompt, skip_vision_check)
        response = cc(model=model, messages=messages, temperature=temperature)
        click.secho(response.choices[0].message['content'], fg='green')
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")



@cli.command()
@click.option('--system_prompt', '-sp', default=None, help='System prompt for the LLM.')
@click.option('--model', '-m', default=configg.model, help='Model name, e.g., provider/model_name.')
@click.option('--temperature', '-t', default=configg.temperature, help='LLM temperature (0-1).')
@click.option('--no_system_prompt', '-nsp', is_flag=True, help='Disable system prompt.')
@click.option('--context', '-c', type=str, default=None, help='Context string or file path.')
@click.option('--skip_vision_check', '-svc', is_flag=True, help='Skip vision model check.')
@click.option('--session_id', '-s', type=str, default=None, help='Session ID for resuming a chat.')
@click.option('--title', '-t', default="Untitled Session", help='Title for the chat session.')
def chatui(system_prompt, model, temperature, no_system_prompt, skip_vision_check, context, session_id, title):
    """Interactive chat interface with Markdown, RAG, and image support."""
    console.print("[cyan]Welcome to ChatUI! Type '/help' for commands.[/cyan]")

    if context:
        raw_context = get_context_from_file(context) or context
        system_prompt = get_prompt("rag") if not no_system_prompt else None
        console.print("[green]Context loaded.[/green]")
    else:
        raw_context = None

    # Reset or load conversation
    messages = reset_conversation(system_prompt, no_system_prompt)
    pending_image = None

    if session_id:
        session = get_chat_history(session_id)
        if session:
            session_id, start_time, title, chat_history = session
            messages = chat_history
            console.print(f"[green]Resumed session '{title}' (ID: {session_id}) from {start_time}, had {len(messages)} turns.[/green]")
        else:
            session_id = None # not exists in db, none will generate random one
            console.print(f"[red]Session ID {session_id} not found. Starting a new session.[/red]")

    # Display initial messages
    show_messages(messages, console)

    try:
        while True:
            user_input = click.prompt(click.style("You 👦", fg="blue"), default="", show_default=False).strip()
            if not user_input:
                continue

            command = filter_command(user_input)
            match command.lower():
                case "/exit"| "/quit":
                    break
                case "/clear":
                    messages = reset_conversation(system_prompt, no_system_prompt)
                    pending_image = None
                    console.print("[yellow]Conversation history cleared.[/yellow]")
                    continue
                case "/clear_context":
                    raw_context=None
                    console.print("[yellow]Context cleared![/yellow]")
                    continue
                case "/help":
                    console.print("[yellow]Commands: /exit, /quit, /clear, /help, /image <path>.[/yellow]")
                    continue
                case "/image":
                    pending_image = handle_image_command(user_input, model, skip_vision_check, console)
                    console.print("Image attached. Add a message") if pending_image else None
                    continue
                case _:
                    pass

            # Use context if available for RAG
            if raw_context:
                formatted_message = get_formatter('rag').format(question=user_input, context=raw_context)
            else:
                formatted_message = user_input

            # Add user input or image + message to the conversation
            if pending_image:
                content = parse_image(image=pending_image, message=formatted_message)
                messages.append({"role": "user", "content": content})
                pending_image = None
            else:
                messages.append({"role": "user", "content": formatted_message})

            # Send messages to LLM and handle the response
            try:
                response = cc(model=model, messages=messages, temperature=temperature)
                assistant_message = response.choices[0].message["content"]
                messages.append({"role": "assistant", "content": assistant_message})
                show_messages([{"role": "assistant", "content": assistant_message}], console)
            except Exception as e:
                console.print(f"[red]Error generating response: {e}[/red]")
    finally:
        session_id=save_chat_history(messages, session_id=session_id, title=title)        
        console.print(f"[cyan]Session saved with ID {session_id} Goodbye![/cyan]")



@cli.command()
@click.option('--delete_session', type=str, required=False, default=None, help='Session ID to delete, or "all" to delete all sessions.')
def history(delete_session):
    """List all saved chat sessions."""
    
    if delete_session:
        delete_chat_session(delete_session)
        console.print(f"[green]Session {delete_session} deleted.[/green]")
        return
        
    sessions = get_chat_history()
    console.print("[cyan]Chat History:[/cyan]\n")
    if sessions:
        for session in sessions:
            session_id, start_time, title, length = session
            console.print(f"[yellow]Session ID:[/yellow] {session_id}\n"
                          f"[yellow]Start Time:[/yellow] {start_time}\n"
                          f"[yellow]Title:[/yellow] {title}\n"
                          f"[yellow]Length:[/yellow] {length} turns\n")
    else:
        console.print("[red]No chat history found.[/red]")
        
        
@cli.command()
@click.argument('key', required=False, default=None)
@click.argument('value', required=False, default=None)
def config(key: str, value: str):
    """
    Configure chat settings.
    Without arguments, displays the current configuration.
    To update, provide a key and a value.
    """
    current_config = load_config()
    config_dict = asdict(current_config)

    if not key:
        # Show current configuration if no arguments are provided
        click.secho("Current configuration:", fg='cyan')
        for field_name, field_value in config_dict.items():
            click.secho(f"  {field_name}: {field_value}", fg='yellow')
        return

    # Validate the provided key
    if key not in config_dict:
        click.secho(f"Invalid configuration key: {key}", fg='red')
        click.secho("Valid keys are:", fg='cyan')
        for field_name in config_dict.keys():
            click.secho(f"  {field_name}", fg='yellow')
        return

    # Validate the provided value against the field's type
    field_type = {f.name: f.type for f in fields(ChatConfig)}[key]
    try:
        # Dynamically cast the value to the correct type
        if field_type == int:
            value = int(value)
        elif field_type == float:
            value = float(value)
        elif field_type == str:
            value = str(value)
        else:
            raise ValueError(f"Unsupported type: {field_type}")
    except ValueError:
        click.secho(f"Invalid value type for {key}. Expected {field_type.__name__}.", fg='red')
        return

    # Update the configuration
    setattr(current_config, key, value)
    save_config(current_config)
    click.secho(f"Configuration updated: {key} = {value}", fg='green')


if __name__ == "__main__":
    cli.add_command(chat)
    cli.add_command(chatui)
    cli.add_command(config)
    cli.add_command(history)
    cli()
