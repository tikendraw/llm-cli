# LLM-cli
A lightweight Command Line Interface (CLI) for interacting with Large Language Models (LLMs) using LiteLLM.


## 💡 Why This Project?
Sometimes network constraints or data limitations make it difficult to access large language models via web interfaces. This CLI provides a lightweight, flexible solution for LLM interactions directly from the terminal.


## 🚀 Features

- **Simple CLI Interface**: Easily chat with different LLMs from your terminal
- **Multiple Chat Modes**:
  - Direct single-message chat
  - Interactive chat UI with markdown rendering
  - Image support for vision-capable models
- **Flexible Configuration**: Customize model, temperature, and system prompts
- **Easy Configuration Management**: Update settings with a simple command

## 🔧 Prerequisites

- Api keys to the llms

## 💾 Installation

1. Via Pip
```bash
pip install llm-cli
```
2. From Repo
```bash
# Clone the repository
git clone https://github.com/tikendraw/llm-cli.git
cd llm-cli

# Install 
pip install .
```

## 🖥️ Usage

### Basic Chat

Send a single message to an LLM:

```bash
llm-cli chat "Hello, how are you?"
```

### Interactive Chat UI

Start an interactive chat session:

```bash
llm-cli chatui
```

### Image Support

Chat with an image:

```bash
llm-cli chatui2 --model openai/gpt-4o-somthing
```

### Configuration

View current configuration:
```bash
llm-cli config
```

Update configuration:
```bash
llm-cli config model "anthropic/claude-3-haiku"
llm-cli config temperature 0.7
```

## 🛠️ Commands

- `chat`: Send a single message
- `chatui`: Interactive chat with markdown rendering
- `chatui2`: Interactive chat with image support
- `config`: Manage CLI configuration


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

