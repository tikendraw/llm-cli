import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

config_dir = os.path.expanduser('~/.llm-cli')
config_dir = Path(config_dir)
config_dir.mkdir(parents=True, exist_ok=True)

config_file = config_dir/'config.json'
history_db = config_dir/'chat_history.db'



@dataclass
class ChatConfig:
    model:str = 'gemini/gemini-1.5-flash'
    temperature:float = 0.2
    max_token_output:int = 8192
    

def save_config(config:ChatConfig=None, config_file:Path=config_file)->None:
    
    if config is None:
        config = ChatConfig()
        
    config = asdict(config)
    with open(config_file, 'w') as f:
        json.dump(config,f)
    
    

def load_config() -> ChatConfig:
    """Load configuration from a JSON file."""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        return ChatConfig(**config_data)
    return ChatConfig()  