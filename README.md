# LLM-cli
A lightweight Command Line Interface (CLI) for interacting with Large Language Models (LLMs) using LiteLLM.


## 💡 Why This Project?
Sometimes network constraints or data limitations make it difficult to access large language models via web interfaces. This CLI provides a lightweight, flexible solution for LLM interactions directly from the terminal.


## 🚀 Features

- **Simple CLI Interface**: Easily chat with different LLMs from your terminal
- **Input**: Pipe inputs or redirect file text.
- **Multiple Chat Modes**:
  - Direct single-message chat
  - Interactive chat UI with markdown rendering
  - Image support for vision-capable models
- **Flexible Configuration**: Customize model, temperature, and system prompts
- **Easy Configuration Management**: Update settings with a simple command
- **Sessions** : Logs chat sessions, can be resumed saved chat later.

## 🔧 Prerequisites

- Api keys to the llms, set api keys as environment variables

## 💾 Installation

1. Via Pip
```bash
pip install llm-to-cli
```
Or 

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

* Send a single message to an LLM:

  ```bash
  llm-cli chat "Hello, how are you?"
  ```
* Pipe input
  ```bash
  echo "what is 34th prime number" | llm-cli chat
  ```
* File redirection
  ```bash
  llm-chat chat < some_file_with_question.txt
  ```

### Interactive Chat UI

* Start an interactive chat session:

  ```bash
  llm-cli chatui
  ```

### Image Support

* Add image
  ```bash
  llm-cli chat --image path/to/image/or/url
  ```

### Configuration

* View current configuration:
  ```bash
  llm-cli config
  ```

* Update configuration:
  ```bash
  llm-cli config model "anthropic/claude-3-haiku"
  llm-cli config temperature 0.7
  ```

## 🛠️ Commands

- `chat`: Send a single message
- `chatui`: Interactive chat 
- `config`: Manage CLI configuration
- `history`: See and manage history

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🖹 License
The MIT one.