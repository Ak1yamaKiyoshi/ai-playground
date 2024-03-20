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
from agents import OpenAIAgent, CoderAgent, DebuggerAgent, ConsolerAgent, NamerAgent, SummaryAgent
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


def code_gen_pipeline(query:str, meta:str, purpose:str, coder_agent:CoderAgent, namer_agent:NamerAgent):
    date = time.strftime("%d.%m.%Y | %H:%M:%S")
    response = coder_agent.invoke(query, "coder", meta, temperature=0.6)
    name_response = namer_agent.invoke(response, "namer", 
                                    meta, temperature=0.3)

    name_response = preprocess_name(name_response)

    code_response = response.split("```")[1].replace("```", "").replace("python\n", "")
    
    formatted_response = code2md_with_meta(meta, purpose, name_response, code_response)

    # debug 
    with open("./debug_history_coder.txt", "w") as f:
        f.write("\n".join([str(entry) + "\n" for entry in coder_agent.history]))

    # Saving 
    if not os.path.isfile(cached_filename):
        with open(cached_filename, "w+") as f:
            f.write(formatted_response)
    else: 
        with open(cached_filename, "a+") as f:
            f.write(formatted_response)

    with open(os.path.join(output_dir, name_response), "w+") as f:
        f.write(code_response)
    
    print(f"Code saved to {name_response}\n Check it and leave feedback. \n Date: {time.strftime('%d.%m.%Y %H:%M:%S')}")
    return {
        "filename": name_response,
        "meta": meta,
        "purpose": purpose,
        "code": code_response,
        "date": date
        
    }


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


def pipeline(query:str, meta:str, purpose:str, always_to_add:str,
             coder_agent: CoderAgent, namer_agent: NamerAgent, summary_agent: SummaryAgent):
    uid = 0
    date_user = time.strftime("%d.%m.%Y | %H:%M:%S")
    summary_history = []
    while True:
        res = code_gen_pipeline(query, meta, purpose, coder_agent, namer_agent)
    
        summary = summary_agent.invoke(res["code"], "summary", meta, temperature=0.5)
        summary_history.append(summary)
        summary_history_str = "\n- YOU ARE BANNED FROM IMPLEMENTING: ".join(summary_history)
        
        if PriceCounter.count_tokens_in_chat(coder_agent.get_history()) > 4000:
            coder_agent.wipe_first_utterance()
        if PriceCounter.count_tokens_in_chat(namer_agent.get_history()) > 4000:
            namer_agent.wipe_first_utterance()
        
        save_as_dialog(res, query, date_user, uid, summary)

        query = f"""
{always_to_add}
- COMMAND YOU MUST EXECUTE:
```
{minput(" > Feedback: ")}
```

- ALREADY TRIED SOLUTIONS COMMANDS:
{summary_history_str}
"""
        query = query.replace("STOP", "")

        date_user = time.strftime("%d.%m.%Y | %H:%M:%S")
        print("Running new query. \n")
        uid += 1


# Will be used as promptp for each iteration
consistent_prompt = """
You are not allowed to use cv trackers or any deeplearning solutions.
Take to attention feedback, and do not reimplement banned solutions.
No yapping and comments.
Using ROI selection are banned too.
```py
import os
from typigng import List
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

first_query = f"""I want to track object on video using lucas kanade algorithm.
Divide algorythm in two stages: 
1. You should implement object detection with background substractors and kalman filter, output of that, when object is deteccted should be bounding boxes, or BBOX in other words. 
2. After that, you need to track that BBOX from previous step with lucas kanade filter. 
3. If BBOX's cound == 0 you should return to the first step 
{consistent_prompt}
"""

coder = CoderAgent()
namer = NamerAgent()
summarizer = SummaryAgent()
pipeline(first_query, "opencv-gen-test", "coder", consistent_prompt, coder, namer, summarizer)
