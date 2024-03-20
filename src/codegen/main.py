from shot_config import ShotConfig
from general_config import Config
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

import tiktoken
import logging
import time
import os
from log_utils import Logger, PriceCounter
from shots import Shot, PredefinedShot, OpenAIClient
from agents import OpenAIAgent, CoderAgent, NamerAgent, SummaryAgent, AnalystAgent
import shutil
load_dotenv()


directory = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(directory, "code_output")
dialog_file = os.path.join(directory, "cached_dialog.md")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(directory, exist_ok=True)
# Clear directory.
for file in os.listdir(output_dir):
    if os.path.isfile(os.path.join(output_dir, file)) and file.endswith(".py"):
        os.remove(os.path.join(output_dir, file))

cached_filename = f"{directory}/cached_responses.md"

def preprocess_name(name:str):
    return name.replace("`", "")

def code2md_with_meta(meta:str, purpose:str, name_response:str, code_response:str):
    date = time.strftime("%d.%m.%Y-%H:%M:%S")
    return f"""# {name_response}
### Date: {date}
### Meta: {meta}
### Purpose: {purpose}
```py
{code_response}
```
"""


def minput(string:str):
    print("\n type `STOP` to stop multiline input.\n" + string)
    contents = []
    running = True
    while running:
        try:
            line = input()
        except EOFError:
            break
        
        if 'STOP' in line:
            contents.append(line)
            running = False
        contents.append(line)
    return "\n".join(contents)


def invoke_coder(query:str, meta:str, purpose:str, coder_agent:CoderAgent):
    retry = 3
    while retry > 0:
        try:
            response = coder_agent.invoke(query, "coder", meta, temperature=0.5, output_dir=output_dir)
            date = response.date
            code = eval(response.response)['code']
            filename =  eval(response.response)['filepath']
            formatted_response = code2md_with_meta(meta, purpose, filename, code)
            retry = 0
        except:
            print("retry...")
            retry -= 1
    if not os.path.isfile(cached_filename):
        with open(cached_filename, "w+") as f:
            f.write(formatted_response)
    else: 
        with open(cached_filename, "a+") as f:
            f.write(formatted_response)

    try: os.makedirs(os.path.join(output_dir, os.path.join(filename.split("/")[:-1])), exist_ok=True)
    except:pass
    try: os.mkdir(os.path.join(output_dir, os.path.join(filename.split("/")[:-1])))
    except:pass
    filename = filename.split("/")[-1]
    with open(os.path.join(output_dir, filename), "w+") as f:
        f.write(code)

    return {
        "filename": filename,
        "meta": meta,
        "purpose": purpose,
        "code": code,
        "date": date   
    }


def invoke_analyst(query:str, meta:str, analyst_agent:AnalystAgent, summary_history: List[str], consistent_requirement:str, code):
    summary_history_str = "\n- ".join(summary_history)
    query = f"""
{consistent_requirement}
```
{minput(" > Feedback: ")}
```
ALREADY TRIED SOLUTIONS OR COMMANDS:
```
{summary_history_str}
```
CODE:
```py
{code}
```
"""

    query = query.replace("STOP", "")
    query = analyst_agent.invoke(query, "analyst", meta, temperature=0.5, output_dir=output_dir).response
    return query 


def save_as_dialog(res, query, date_user:str, uid:int, summary:str):
    date = time.strftime("%d.%m.%Y | %H:%M:%S")
    
    utterance_begin = f"\n---\n```{date}```\n```utterance {uid}```\n"
    user_query = f"""### User
#### {date_user}
{query}
"""

    ai_response = f"""### AI
#### {res['date']}
#### Filename: {res['filename']}
#### Summary: {summary}
#### Code: 
```py
{res['code']}
```
"""
    with open(dialog_file, "a+") as f:
        f.write(utterance_begin)
        f.write(user_query)
        f.write(ai_response)


def invoke_summarizer(res, meta:str, summary_list:List[str], summarizer_agent:SummaryAgent):
    summary_list.append( summarizer_agent.invoke(res["code"], "summary", meta, temperature=0.5, output_dir=output_dir).response)
    return summary_list[-1]

def pipeline(meta:str, purpose:str, always_to_add:str,
             coder_agent: CoderAgent, namer_agent: NamerAgent, summary_agent: SummaryAgent,
             analyst_agent: AnalystAgent):
    uid = 0
    summary_history = []
    code = ""
    query = ""
    while True:
        date_user = time.strftime("%d.%m.%Y | %H:%M:%S")
        query =   invoke_analyst(query, meta, analyst_agent, summary_history, always_to_add, code)
        res    =  invoke_coder(query, meta, purpose, coder_agent)
        summary = invoke_summarizer(res, meta, summary_history, summary_agent)
        code = res['code']
        save_as_dialog(res, query, date_user, uid, summary)
    
        
        if PriceCounter.count_tokens_in_chat(coder_agent.get_history()) > 4000:
            coder_agent.wipe_first_utterance()
        if PriceCounter.count_tokens_in_chat(namer_agent.get_history()) > 4000:
            namer_agent.wipe_first_utterance()
        if PriceCounter.count_tokens_in_chat(summary_agent.get_history()) > 4000:
            summary_agent.wipe_first_utterance()
        if PriceCounter.count_tokens_in_chat(analyst_agent.get_history()) > 9000:
            analyst_agent.wipe_first_utterance()
        
        print(f"Code saved to {res['filename']}\n Check it and leave feedback. \n Date: {time.strftime('%d.%m.%Y %H:%M:%S')}")
        
        
        print("Running new query. \n")
        uid += 1



# Will be added to prompt for each iteration
consistent_prompt = """
IMPLEMENT EVERYTHING: 

You are not allowed to use cv trackers or any deeplearning solutions.
Take to attention feedback, and do not reimplement banned solutions.
No yapping and comments.
Using ROI selection are banned too.
Use this as starting code:
```py
import os
from typing import List
import traceback
if __name__ == "__main__":
    videos=[]
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            videos.append(os.path.join("./assets", path))
    for video in sorted(videos):
        try: track(video)
        except: traceback.print_exc()
```
"""


coder = CoderAgent()
namer = NamerAgent()
summarizer = SummaryAgent()
analyst = AnalystAgent()
pipeline("opencv-gen-test", "coder", consistent_prompt, coder, namer, summarizer, analyst)




""" 
pipeline:
do:
  -- Decisionmaking: 
  Analyst outputs [requirements list, user task (feedback)]
  Decisionmaker outputs [   
    action: wipe out memory for coder | analyst | do not wipe out memory 
    idea: stick to the current solution, or new solution general idea
    details: plan what to do and what to update
  ]
  -- File managing: 
  Coder outputs [code, path_filename]
    Summarizer outputs what coder did
  User runs the thing.
while (money on account)
  
  
  чел який буде по директоріям все розкидувати -> Анатолій
"""
# Писати diff файли замість повних файліх
