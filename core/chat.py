import base64
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import litellm
from litellm import completion

from .config import config_dir


def chat(*args, **kwargs):
    return completion(*args, **kwargs)


def stream_chat(*args, **kwargs):
    kwargs["stream"] = True
    return completion(*args, **kwargs)


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue

            text_value = item.get("text", {}).get("value") if isinstance(item.get("text"), dict) else None
            if isinstance(text_value, str):
                parts.append(text_value)

        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("content")
        if text is not None:
            return _extract_content_text(text)

        text = content.get("text")
        if isinstance(text, str):
            return text

    return ""


def extract_chunk_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None and isinstance(choice, dict):
        delta = choice.get("delta")
    if delta is not None:
        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")
        return _extract_content_text(content)

    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return _extract_content_text(content)


def collect_stream_response(
    response_stream: Iterable[Any], on_chunk: Callable[[str], None] | None = None
) -> str:
    chunks: list[str] = []

    for chunk in response_stream:
        text = extract_chunk_text(chunk)
        if not text:
            continue
        chunks.append(text)
        if on_chunk is not None:
            on_chunk(text)

    return "".join(chunks)

def is_vision_llm(model:str)->bool:
    return litellm.supports_vision(model)

def encode_image(image_path: str |Path) -> str:
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def is_url(path:Path)->bool:
    return path.as_posix().startswith("http")

def parse_image(image:str|Path, message:str=None)->list[dict]:
    if isinstance(image, str):
        image = Path(image)

    if url:=is_url(image):
        base_image = image.as_posix()
    else:
        try:
            base_image = encode_image(image)
        except FileNotFoundError as e:
            return []

    message_part =[]
    if message:
        message_part.append({
                                "type": "text",
                                "text": message
                            })
    if base_image:
        message_part.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": base_image if url else f"data:image/jpeg;base64,{base_image}"
                                }
                            }
        )
        
    return message_part
    


def random_name() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}-{random.randint(100000000, 9999999999)}"

def unparse_image(message: List[Dict], config_dir: str = config_dir/'attached_images') -> Tuple[str, str]:
    """
    Get user message and image file path from a parsed message.
    
    Args:
        message (list[dict]): Parsed message containing text and/or image data.
        config_dir (str): Directory to save decoded images.
        
    Returns:
        Tuple[str, str]: A tuple containing the user message and the saved image file path.
    """
    user_message = None
    image_path = None

    # Ensure config_dir exists
    os.makedirs(config_dir, exist_ok=True)

    for part in message:
        if part["type"] == "text":
            user_message = part["text"]
        elif part["type"] == "image_url":
            image_data = part["image_url"]["url"]
            if image_data.startswith("data:image/jpeg;base64,"):
                # Decode base64 image and save it
                image_data = image_data.split(",", 1)[1]  # Remove the `data:image/jpeg;base64,` prefix
                decoded_image = base64.b64decode(image_data)
                image_path = Path(config_dir) / f"{random_name()}.jpg"
                with open(image_path, "wb") as f:
                    f.write(decoded_image)
            else:
                image_path = image_data

    return user_message, str(image_path) if image_path else None
