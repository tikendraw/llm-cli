_config = None

def get_config():
    global _config
    if _config is None:
        from core.config import load_config
        _config = load_config()
    return _config
