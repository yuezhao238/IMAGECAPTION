# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train import train
import os

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = '6b6ae714ca6908898fec2f0198691c5e2a52b7f7'
    os.environ["WANDB_MODE"] = "offline"
    train()
