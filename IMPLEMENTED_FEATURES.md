# Implemented Features Guide

This document covers the features that are implemented and usable in the current `llm-cli` codebase.

It focuses on the commands that are wired into the CLI and have clear, supported behavior:

- `chat`
- `chatui`
- `config`
- `history`
- `agent` (optional extra)

## Installation

Install from the repository root:

```bash
pip install -e .
```

After installation, verify the CLI is available:

```bash
llm-cli --help
```

## Requirements

`llm-cli` uses LiteLLM, so you need API credentials for the model provider you want to use.

The default model is:

```bash
gemini/gemini-3-flash-preview
```

Set the matching provider API key in your environment before using `chat`, `chatui`, or `agent`.

Example:

```bash
export GEMINI_API_KEY=your_key_here
```

## Where Data Is Stored

The CLI stores its local files in:

```bash
~/.llm-cli/
```

Files created there:

- `config.json`: saved configuration
- `chat_history.db`: SQLite database used by `chatui` and `history`
- `attached_images/`: images restored from saved history when needed

## 1. `chat`

Use `chat` for one-shot prompts from the terminal.

### Basic usage

```bash
llm-cli chat "Hello"
```

### Supported input methods

Pass the message directly:

```bash
llm-cli chat "Explain recursion simply"
```

Read from a file:

```bash
llm-cli chat --file question.txt
```

Pipe input:

```bash
echo "Summarize this topic" | llm-cli chat
```

Redirect stdin:

```bash
llm-cli chat < question.txt
```

### Useful options

Override the model:

```bash
llm-cli chat -m openai/gpt-4o-mini "Write a haiku"
```

Set temperature:

```bash
llm-cli chat -t 0.7 "Give me three creative names"
```

Add a custom system prompt:

```bash
llm-cli chat -s "You are a concise assistant." "Explain TCP"
```

Disable the system prompt:

```bash
llm-cli chat --no_system_prompt "Answer plainly"
```

### Context support

You can pass extra context as a string or file path:

```bash
llm-cli chat -c notes.txt "Summarize the important points"
```

If the value passed to `--context` is a real file path, the file contents are loaded and used as context.

### Image input

You can attach an image for vision-capable models:

```bash
llm-cli chat --image ./photo.jpg -m gemini/gemini-3-flash-preview "What is in this image?"
```

If the selected model is not recognized as vision-capable, the command will stop unless you pass:

```bash
--skip_vision_check
```

## 2. `chatui`

Use `chatui` for an interactive conversation in the terminal.

### Start a session

```bash
llm-cli chatui
```

### Start with options

Use a custom model:

```bash
llm-cli chatui -m openai/gpt-4o-mini
```

Use a system prompt:

```bash
llm-cli chatui --system_prompt "You are a coding tutor."
```

Start with context:

```bash
llm-cli chatui -c project_notes.txt
```

Resume an earlier session:

```bash
llm-cli chatui --session_id YOUR_SESSION_ID
```

Set a session title:

```bash
llm-cli chatui --title "Project Discussion"
```

### In-chat commands

Supported interactive commands:

- `/help`
- `/exit`
- `/quit`
- `/clear`
- `/clear_context`
- `/image <path>`

### What `chatui` saves

When you exit `chatui`, the conversation is saved to `~/.llm-cli/chat_history.db`.

That saved history can later be:

- listed with `llm-cli history`
- reopened with `llm-cli chatui --session_id ...`

## 3. `config`

Use `config` to inspect or update default settings.

### Show current config

```bash
llm-cli config
```

### Update values

Set the default model:

```bash
llm-cli config model gemini/gemini-3-flash-preview
```

Set the default temperature:

```bash
llm-cli config temperature 0.2
```

Set max output tokens:

```bash
llm-cli config max_token_output 8192
```

### Supported config keys

- `model`
- `temperature`
- `max_token_output`

On first config load, the CLI creates:

```bash
~/.llm-cli/config.json
```

## 4. `history`

Use `history` to inspect or delete saved chat sessions.

### List sessions

```bash
llm-cli history
```

The command shows:

- session ID
- start time
- title
- conversation length

### Delete one session

```bash
llm-cli history --delete_session YOUR_SESSION_ID
```

### Delete all sessions

```bash
llm-cli history --delete_session all
```

## 5. `agent` (optional)

The `agent` command is implemented, but it requires the optional `agents` extra.

Install it with:

```bash
pip install -e '.[agents]'
```

### Basic usage

```bash
llm-cli agent "Write a hello world program"
```

Read from a file:

```bash
llm-cli agent --file task.txt
```

Pipe input:

```bash
echo "Research this topic" | llm-cli agent
```

Override the model:

```bash
llm-cli agent -m openai/gpt-4o-mini "Plan a small CLI tool"
```

### What the agent feature includes

The current implementation builds a manager agent with sub-agents for:

- simple web search
- Python/code tasks
- Bash tasks

## Not Covered Here

This guide intentionally does not document features that are not fully established in the current package workflow.

In particular, the `audio` command exists in the CLI, but it depends on additional runtime packages and external tools that are not declared in the main installation path of this repository. Because of that, it is not included in this "implemented and ready-to-use" guide.

## Quick Start

If you just want the shortest path:

```bash
pip install -e .
export GEMINI_API_KEY=your_key_here
llm-cli config
llm-cli chat "Hello"
llm-cli chatui
```
