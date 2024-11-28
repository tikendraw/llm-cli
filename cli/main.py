import os
from dataclasses import asdict, fields

import click
from rich.console import Console
from rich.markdown import Markdown

from cli.utils import delete_chat_session, get_chat_history, init_db, save_chat_history
from core.chat import chat as cc
from core.chat import is_vision_llm, parse_image, unparse_image
from core.config import ChatConfig, load_config, save_config
from core.prompt import system_prompt_cot

sys_p = system_prompt_cot
console = Console()
configg = load_config()

init_db()


@click.group
def cli():
    """ llm-cli application """
    pass



@cli.command()
@click.argument('message', type=str)
@click.option('--system_prompt', '-s', default=sys_p, type=str, help='system prompt to the llm')
@click.option('--model', '-m', default=configg.model, help='model name e.g.: provider/model_name')
@click.option('--temperature', '-t', default=configg.temperature, help='float value between 0 and 1, lower value means more deterministic, higher value means more creative')
@click.option('--image', type=click.Path(exists=True), default=None, help='file path or url to an image')
@click.option('--no_system_prompt', is_flag=True, help='disable system prompt')
@click.option('--skip_vision_check', is_flag=True, help='skip vision check')
def chat(model, message, temperature, system_prompt, image, no_system_prompt, skip_vision_check):
    messages = []
    if system_prompt and not no_system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    if image:
        if is_vision_llm(model) or skip_vision_check:
            content = parse_image(image=image, message=message)
            messages.append({'role' : 'user', 'content' : content})
        else:
            console.print(f'[red]{model} is not a vision model(according to litellm).[/red]')
            return
    else:            
        messages.append({"role": "user", "content": message})
    
    # print('model: ', model)
    # print(f'gave this: {messages}')
    
    response = cc(model=model, messages=messages, temperature=temperature)
    click.secho(response.choices[0].message['content'], fg='green')
    

def print_markdown(message: str) -> None:
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
        print_markdown(content)


@cli.command()
@click.option('--system_prompt', '-s', default=sys_p, type=str, help='System prompt to the LLM')
@click.option('--model', '-m', default=configg.model, help='Model name, e.g., provider/model_name')
@click.option('--temperature', '-t', default=configg.temperature, help=f'Temperature for the LLM, defaults to {configg.temperature}')
@click.option('--no_system_prompt', is_flag=True, help='Disable system prompt')
@click.option('--skip_vision_check', is_flag=True, help="Skip vision model check, (useful when litellm doesn't see the model as vision model)")
@click.option('--session_id', type=str, default=None, help='Session ID to continue')
@click.option('--title', '-t', default="Untitled Session", help='Title/description for the session')
def chatui(system_prompt, model, temperature, no_system_prompt, skip_vision_check, session_id, title):
    """Interactive chat interface with markdown rendering and image support."""
    console.print("[cyan]Welcome to ChatUI![/cyan]")
    console.print("[cyan]Type '/exit' to quit, '/clear' to reset conversation, '/help' for help, or '/image <path>' to add an image.[/cyan]\n")

    messages = []
    pending_image = None

    if system_prompt and not no_system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if session_id:
            session = get_chat_history(session_id)
            if session:
                session_id, start_time, session_title, chat_history = session
                messages = chat_history
                console.print(f"[green]Continuing session '{session_title}' (ID: {session_id}) from {start_time}[/green]")
            else:
                console.print(f"[red]Session ID {session_id} not found. Starting a new session.[/red]")
    else:
        session_id = None


    show_messages(messages, console)

    while True:
        user_input = click.prompt(click.style("You 👦", fg="blue"), default="", show_default=False).strip()

        if user_input.lower() in {"/exit", "/quit"}:
            session_id = save_chat_history(messages, session_id=session_id, title=title)
            console.print(f"[cyan]Session saved with ID {session_id}. Goodbye![/cyan]")
            break
        elif user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}] if not no_system_prompt else []
            pending_image = None
            console.print("[yellow]Conversation history cleared.[/yellow]\n")
            continue
        elif user_input.lower() == "/help":
            console.print("[yellow]Available commands:[/yellow]\n"
                          "[bold yellow]/help[/bold yellow] - Show help message\n"
                          "[bold yellow]/exit or /quit[/bold yellow] - Exit the chat\n"
                          "[bold yellow]/clear[/bold yellow] - Reset conversation history\n"
                          "[bold yellow]/image <path>[/bold yellow] - Add an image to the conversation\n")
            continue

        if user_input.startswith("/image"):
            try:
                _, image_path = user_input.split(maxsplit=1)
                if not os.path.exists(image_path):
                    console.print(f"[red]Image file '{image_path}' not found![/red]")
                    continue
                if is_vision_llm(model) or skip_vision_check:
                    pending_image = image_path
                    console.print(f"[green]Image '{image_path}' added. Enter a message with the image.[/green]")
                else:
                    console.print(f"[red]{model} is not a vision model.[/red]")
                    pending_image = None
            except ValueError:
                console.print("[red]Please provide a valid image path using '/image <path>'.[/red]")
            continue

        if pending_image:
            content = parse_image(image=pending_image, message=user_input)
            messages.append({"role": "user", "content": content})
            pending_image = None
        else:
            messages.append({"role": "user", "content": user_input})

        response = cc(model=model, messages=messages, temperature=temperature)
        llm_message = response.choices[0].message["content"]
        messages.append({"role": "assistant", "content": llm_message})

        show_messages([{"role": "assistant", "content": llm_message}], console)




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
