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

class AgentResponse:
    def __init__(self, response:str, purpose:str) -> None:
        self.response = response
        self.date = time.strftime("%d.%m.%Y | %H:%M:%S")
        self.purpose = purpose
        self.dict = {
            "response": response,
            "date": self.date,
            "purpose": purpose
        }
        
        
    def __str__(self) -> str:
        return self.response
    
    def __repr__(self) -> str:
        return self.response


class OpenAIAgent(OpenAIClient):
    Avaivable = PredefinedShot.Avaiavable
    
    def __init__(self, agent_type) -> None:
        super().__init__(PredefinedShot(agent_type))
    
    def invoke(self, query:str, purpose:str, meta:str, temperature:int=0.5, output_dir:str="./"):
        response = self.completion(query, purpose, meta, temperature, output_dir)
        return AgentResponse(response, purpose)

        
class CoderAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.code_generate)

class NamerAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.namer)

class SummaryAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.summary)

class AnalystAgent(OpenAIAgent):
    def __init__(self) -> None:
        super().__init__(OpenAIAgent.Avaivable.analyst)