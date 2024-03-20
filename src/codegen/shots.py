from shot_config import ShotConfig
from general_config import Config
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from pprint import pprint
import logging
import time
from log_utils import Logger, PriceCounter

import os

load_dotenv()


class Shot:
    def __init__(self, shots: List[Tuple[str, str]], system_message:str, prompt:str):
        self.shots = shots
        self.system_message = system_message
        self.prompt = prompt

    def __wrap(self, role:str, query:str) -> str:
        return {"role": role, "content": query}


    def messages(self, query: str, history:List[Dict[str, str]]=[]) -> List[Dict[str, str]]:
        message_shots = []
        for shot in self.shots:
            message_shots += [self.__wrap("user", shot[0]),
            self.__wrap("assistant", shot[1])]
        
        messages = [
            self.__wrap("system", self.system_message),
            *message_shots,
            *history,
            self.__wrap("user", query)
        ]
        return messages


class PredefinedShot(Shot):
    class Avaiavable:
        code_generate = "gen"
        code_debug = "debug"
        console_action = "action"
        namer = "namer"
        summary ="summary"
    
    def __init__(self, shot_type:str) -> None:
        if shot_type == self.Avaiavable.code_generate:
            super().__init__(ShotConfig.Coder.shots,   ShotConfig.Coder.system_message, ShotConfig.Coder.prompt)
        elif shot_type == self.Avaiavable.code_debug:
            super().__init__(ShotConfig.Debugger.shots, ShotConfig.Debugger.system_message, ShotConfig.Debugger.prompt)
        elif shot_type == self.Avaiavable.console_action:
            super().__init__(ShotConfig.Consoler.shots, ShotConfig.Consoler.system_message, ShotConfig.Consoler.prompt)
        elif shot_type == self.Avaiavable.namer:
            super().__init__(ShotConfig.Namer.shots, ShotConfig.Namer.system_message, ShotConfig.Namer.prompt)
        elif shot_type == self.Avaiavable.summary:
            super().__init__(ShotConfig.Summary.shots, ShotConfig.Summary.system_message, ShotConfig.Summary.prompt)


class OpenAIClient:
    def __init__(self, shot:Shot) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo-0125"
        self.shot = shot
        self.history:Dict[str, str] = []

    def append_history(self, query:str, response:str):
        self.history += [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]

    def set_history(self, history:List[Dict[str, str]]):
        self.history = history
    
    def wipe_history(self):
        self.history = []
    
    def wipe_first_utterance(self):
        self.history = self.history[2:]
    
    def get_history(self):
        return self.history

    def completion(self, query: str, purpose:str, meta:str = "completion", temperature=0) -> str:
        messages = self.shot.messages(query, history=self.history)
        try:
            in_tokens = PriceCounter.count_tokens_in_chat(messages)
        except:
            print(messages)
            exit()
        start = time.time()
        response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
        ).choices[0].message.content
        end = time.time()
        
        out_tokens = PriceCounter.count_tokens_in_text(response)
        time_taken = round(end-start, 1)
        query_tokens = PriceCounter.count_tokens_in_text(query)
        
        total_price, price = PriceCounter.get_instance().add_transaction(out_tokens, in_tokens, self.model)
        log_message = f" in: {in_tokens:4} | query: {query_tokens:4} | out: {out_tokens:4} | price ${round(price, 4):6} | total_price ${round(total_price, 4):6}"
        
        self.append_history(query, response)
        
        Logger.log(purpose, log_message, purpose, time_taken, meta)
        
        return response

