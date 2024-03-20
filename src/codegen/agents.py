from shot_config import ShotConfig
from general_config import Config
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import logging
import time
import sys
from io import StringIO
from log_utils import Logger, PriceCounter
from shots import Shot, PredefinedShot, OpenAIClient

import os

load_dotenv()


class OpenAIAgent(OpenAIClient):
    Avaivable = PredefinedShot.Avaiavable
    
    def __init__(self, agent_type) -> None:
        super().__init__(PredefinedShot(agent_type))
    
    def invoke(self, query:str, purpose:str, meta:str, temperature:int=0.5):
        response = self.completion(query, purpose, meta, temperature)
        return response

        
class CoderAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.code_generate)

class NamerAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.namer)

class SummaryAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.summary)

class DebuggerAgent(OpenAIAgent):
    def __init__(self, code:str) -> None:
        super().__init__(OpenAIAgent.Avaivable.code_debug)
        self.code = code
        self.max_retries = 3


    def get_errors(self):
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            eval(self.code)
        finally:
            stdout_output = sys.stdout.getvalue()
            stderr_output = sys.stderr.getvalue()
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig

        return stdout_output, stderr_output


    def save_code_history(self):
        # ShotConfig.Debugger.history_dir
        # Todo save code to history file in md format 
        pass 


    def invoke(self):
        retries = 0
        while self.max_retries > retries:
            self.save_code_history()
            stdout, stderr = self.get_errors()
            if stderr:
                retries += 1
                # Todo: add metadata as filetree 
                self.code = self.completion(self.code + "\n" + stderr + stdout)
            else: break
        return self.code


class ConsolerAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.console_action)


# Add to config input dir