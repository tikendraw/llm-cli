import os
from dataclasses import asdict, fields

import click
from rich.console import Console
from rich.markdown import Markdown

from core.chat import chat as cc
from core.chat import is_vision_llm, parse_image
from core.config import ChatConfig, load_config, save_config
from core.prompt import system_prompt as sp
from core.prompt import system_prompt_cot

sys_p = system_prompt_cot
console = Console()
configg = load_config()

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
    

@cli.command()
@click.option('--system_prompt', '-s', default=sys_p, type=str, help='system prompt to the llm')
@click.option('--model', '-m', default=configg.model, help='model name e.g.: provider/model_name')
@click.option('--temperature', '-t', default=configg.temperature, help='temperature to the llm')
def chatui(model, system_prompt, temperature):
    """Interactive chat interface with markdown rendering."""
    messages = [{"role": "system", "content": system_prompt}]
    
    # Greet the user
    console.print("[cyan]Chat session started! Type '/exit' to quit. Type '/help' for commands.[/cyan]\n")
    
    while True:
        user_input = click.prompt(click.style("You 👦", fg='blue'), default="", show_default=False)
        
        match user_input.lower():
            case "/exit" | "/quit":
                console.print("[cyan]Ending chat session. Goodbye![/cyan]")
                break
            case "/help":
                console.print("[yellow]Available commands:[/yellow]\n"
                              "[bold yellow]/help[/bold yellow] - Show this help message\n"
                              "[bold yellow]/exit or /quit[/bold yellow] - End the chat session\n"
                              "[yellow]Type anything else to chat with the assistant.[/yellow]\n")
                continue
            case "/clear":
                messages = [{"role": "system", "content": system_prompt}]
                console.print("[yellow]Conversation history cleared.[/yellow]\n")
                continue

            case _:
                pass

        messages.append({"role": "user", "content": user_input})
        
        response = cc(model=model, messages=messages, temperature=temperature)
        llm_message = response.choices[0].message['content']
        messages.append({"role": "assistant", "content": llm_message})
        
        console.print("[green]LLM 🤖:[/green]")
        try:
            md = Markdown(llm_message)
            console.print(md)
        except Exception:
            console.print(llm_message)
        console.print("\n")

@cli.command()
@click.option('--system_prompt', '-s', default=sys_p, type=str, help='System prompt to the LLM')
@click.option('--model', '-m', default=configg.model, help='Model name, e.g., provider/model_name')
@click.option('--temperature', '-t', default=configg.temperature, help='Temperature for the LLM')
@click.option('--no_system_prompt', is_flag=True, help='Disable system prompt')
@click.option('--skip_vision_check', is_flag=True, help='Skip vision model check')
def chatui2(system_prompt, model, temperature, no_system_prompt, skip_vision_check):
    """Interactive chat interface with markdown rendering and image support."""
    from rich.console import Console
    from rich.markdown import Markdown
    import os

    console = Console()
    messages = []
    pending_image = None  # To store the image path temporarily

    # Add system prompt if not disabled
    if system_prompt and not no_system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Welcome message
    console.print("[cyan]Welcome to ChatUI2![/cyan]")
    console.print("[cyan]Type '/exit' to quit, '/clear' to reset conversation, '/help' for help, or '/image <path>' to add an image.[/cyan]\n")

    while True:
        # Prompt user for input
        user_input = click.prompt(click.style("You 👦", fg='blue'), default="", show_default=False).strip()

        # Handle special commands
        match user_input.lower():
            case "/exit" | "/quit":
                console.print("[cyan]Exiting ChatUI. Goodbye![/cyan]")
                break
            case "/help":
                console.print("[yellow]Available commands:[/yellow]\n"
                              "[bold yellow]/help[/bold yellow] - Show help message\n"
                              "[bold yellow]/exit or /quit[/bold yellow] - Exit the chat\n"
                              "[bold yellow]/clear[/bold yellow] - Reset conversation history\n"
                              "[bold yellow]/image <path>[/bold yellow] - Add an image to the conversation\n"
                              "[yellow]Type a message to chat or use the commands above.[/yellow]\n")
                continue
            case "/clear":
                messages = [{"role": "system", "content": system_prompt}] if not no_system_prompt else []
                pending_image = None  # Reset image state
                console.print("[yellow]Conversation history cleared.[/yellow]\n")
                continue

        # Handle image command
        if user_input.startswith("/image"):
            try:
                _, image_path = user_input.split(maxsplit=1)
                if not os.path.exists(image_path):
                    console.print(f"[red]Image file '{image_path}' not found! Please check the path and try again.[/red]")
                    continue

                if is_vision_llm(model) or skip_vision_check:
                    pending_image = image_path
                    console.print(f"[green]Image '{image_path}' added. Enter a message with the image.[/green]")
                else:
                    console.print(f"[red]{model} is not a vision model (according to litellm).[/red]")
                    pending_image = None
            except ValueError:
                console.print("[red]Please provide a valid image path using '/image <path>'.[/red]")
            continue

        if pending_image:
            content = parse_image(image=pending_image, message=user_input)
            messages.append({"role": "user", "content": content})
            pending_image = None  # Reset the pending image state
        else:
            messages.append({"role": "user", "content": user_input})

        # Get LLM response
        response = cc(model=model, messages=messages, temperature=temperature)
        llm_message = response.choices[0].message['content']
        messages.append({"role": "assistant", "content": llm_message})

        # Display LLM response
        console.print("[green]LLM 🤖:[/green]")
        try:
            md = Markdown(llm_message)
            console.print(md)
        except Exception:
            console.print(llm_message)

        console.print("\n")  

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
    cli.add_command(chatui2)
    cli.add_command(config)
    cli()
