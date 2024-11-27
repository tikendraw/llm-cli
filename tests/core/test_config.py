from core.config import ChatConfig, load_config, save_config

config = ChatConfig(model='me', temperature=.9, max_token_output=333)

save_config(config)

print(load_config())