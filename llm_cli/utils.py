import os
import sys
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.markdown import Markdown

from core.chat import is_vision_llm, parse_image, unparse_image


def print_markdown(message: str, console: Console) -> None:
    """Render a message as Markdown or plain text."""
    try:
        md = Markdown(message)
        console.print(md)
    except Exception:
        console.print(message)
    console.print("\n")


def show_messages(messages: list[dict], console: Console, show_system_prompt: bool = False) -> None:
    """Display chat messages in the console."""
    for message in messages:
        role = message['role']
        
        if role == 'system' and not show_system_prompt:
            continue

        content = message['content']
        image_path = None
        if isinstance(content, list):
            content, image_path = unparse_image(content)



        if role == 'user':
            console.print("[blue]You 👦:[/blue]")
        elif role == 'assistant':
            console.print("[green]LLM 🤖:[/green]")
        elif role == 'system':
            console.print("[cyan]System 🤖:[/cyan]")
        else:
            console.print("[yellow]Unknown Role:[/yellow]")

        if image_path:
            console.print(f'[yellow]Image is saved here: file:///{image_path}[/yellow]')
            
        print_markdown(content, console)


def get_message_or_stdin(message: str, file: Optional[str] = None) -> str:
    """Get message from arguments, stdin pipe, or file."""
    if file:
        return get_input_from_file(file)
    if not message:
        if not sys.stdin.isatty():  # This handles both pipe (|) and redirection (<)
            return sys.stdin.read().strip()
    return message


def get_context_from_file(path: str) -> Optional[str]:
    """Retrieve file content if the provided path exists."""
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8") if file_path.exists() else None


def prepare_messages(message: str, system_prompt: Optional[str], image: Optional[str], model: str, 
                     context: Optional[str], no_system_prompt: bool, skip_vision_check: bool) -> list:
    """Prepare messages for LLM API call."""
    messages = []
    if system_prompt and not no_system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image:
        if is_vision_llm(model) or skip_vision_check:
            messages.append({'role': 'user', 'content': parse_image(image=image, message=message)})
        else:
            raise ValueError(f"{model} is not a vision-enabled model.")
    else:
        messages.append({"role": "user", "content": message})
    return messages


def reset_conversation(system_prompt: Optional[str], no_system_prompt: bool) -> list:
    """Reset conversation history with or without a system prompt."""
    return [{"role": "system", "content": system_prompt}] if system_prompt and not no_system_prompt else []


def handle_image_command(command: str, model: str, skip_vision_check: bool, console:Console) -> Optional[str]:
    """Handle image command input and validate the path."""
    try:
        _, image_path = command.split(maxsplit=1)
        image_path = Path(image_path)
        if not image_path.exists():
            console.print(f"[red]Image file '{image_path}' not found![/red]")
            return None
        if not is_vision_llm(model) and not skip_vision_check:
            console.print(f"[red]{model} is not a vision-enabled model.[/red]")
            return None
        return image_path
    except ValueError:
        console.print("[red]Invalid image command. Use '/image <path>'.[/red]")
        return None


def filter_command(x:str)->Optional[str]:
    x = x.split(maxsplit=1)
    if x:
        return x[0]
    return

def get_input_from_file(file_path: str) -> Optional[str]:
    """Read content from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        raise click.UsageError(f"Failed to read file: {e}")
