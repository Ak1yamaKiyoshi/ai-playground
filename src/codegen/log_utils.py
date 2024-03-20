from shot_config import ShotConfig
from general_config import Config
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import logging
import time

import os

load_dotenv()



class Logger:
    @staticmethod
    def log(purpose:str, message:str, file:str, time_taken_seconds:float, meta:str):
        date = time.strftime("%d.%m.%Y-%H:%M:%S")
        
        try:os.mkdir(Config.log_dir)
        except:pass
        
        message = f"[{date}] [{time_taken_seconds:5}s] {message} : [{meta}] \n"
        with open(os.path.join(Config.log_dir, file), 'a', encoding='utf-8') as f:
            f.write(message)
        logging.info(message)


class PriceCounter:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.total_price = 0

    def reset(self):
        self.total_price = 0

    def get_total_price(self):
        return self.total_price

    def add_transaction(self, output_tokens: int, input_tokens: int, model: str):
        if model not in list(Config.pricing.keys()):
            return

        output_price = output_tokens * Config.pricing[model]["output_price"] / 1000
        input_price = input_tokens * Config.pricing[model]["input_price"] / 1000
        current = output_price + input_price

        self.total_price += current
        return self.total_price, current


    @classmethod
    def count_tokens_in_text(cls, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return num_tokens


    @classmethod
    def count_tokens_in_chat(cls, messages: List[Dict[str, str]]) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
        return num_tokens

