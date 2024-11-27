import base64
from pathlib import Path

import litellm
from litellm import completion


def chat(*args, **kwargs):
    return completion(*args, **kwargs)

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

def parse_image(image:str|Path, message:str=None)->list:
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
    
