# Changelog

All notable user-facing changes to `llm-cli` are documented here.

## 2026-04-14
- Moved from poetry to UV for packaging and distribution, simplifying the project structure and improving build times.
- Updated dependencies, including a major version bump for `litellm` to 1.83.7, which includes important bug fixes and performance improvements.

## 2026-04-12

### Added

- Streaming response support for `llm-cli chat`, so output now appears token-by-token instead of waiting for the full response.
- Streaming response support for `llm-cli chatui`, making interactive conversations feel faster and more natural.
- Stream assembly helpers in the chat layer so streamed output still gets saved as a complete assistant message in session history.
- Test coverage for streaming chunk handling and CLI integration.

## Previously Implemented

### Core chat features

- `chat` command for one-shot prompts from direct arguments, files, pipes, or redirected stdin.
- Model overrides, temperature overrides, and optional custom system prompts.
- Context injection through `--context`, including loading context from a file path.

### Terminal-aware context

- tmux pane history capture with `--pane-history` and `--pane-target` in `chat`.
- `/pane <n> [pane_target]` command inside `chatui` to pull recent terminal commands and output into the active conversation.
- RAG-style prompt formatting when extra context is attached.

### Interactive chat features

- `chatui` interactive terminal interface.
- Session resume with `--session_id`.
- Session titles with `--title`.
- Automatic chat history persistence in SQLite.
- History reset and context reset commands inside the interactive UI.

### Multimodal support

- Image attachment support in `chat` with `--image`.
- In-chat image attachment in `chatui` through `/image <path>`.
- Vision model capability checks with optional `--skip_vision_check`.

### Project configuration and history

- `config` command for viewing and updating saved defaults like model and temperature.
- `history` command for listing saved sessions.
- History deletion support for a single session or all sessions.

### Audio tooling

- `audio` command for transcription and translation workflows.
- Audio preprocessing and chunking support for larger files.
