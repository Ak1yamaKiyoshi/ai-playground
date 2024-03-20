

class ShotConfig:
    class Coder:
        output_keys = ["filepath", "code"]
        system_message = """You are senior python coder, and you been given task what to do. You need to implement the task and provide it in special structure:
{
    "filepath": filename and path to file from root directory of the project,
    "code": your_code_here,
}
""" 
        prompt = """In given format implement user reqirements in code fully. Implement everything and provide most fullfiled result.    No yapping.
Format:
{
    "filepath": filename and path to file from root directory of the project,
    "code": your_code_here,
}
Requirements:
"""
        shots = [
# in 0
('''
Implement fibonacci sequence as short as possible.
''',
# out 0 
'''{
    filepath: "run_fib_sequence.py",
    "code": """
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a
"""
}
'''), ('''
Implement training for binary classification bert-uncased model using transformers library from directory `./src/model.py`
Take IMDBdataset class for dataset: it has __iter__, __len__ and __getitem__ methods and as inputs it has any iterable dataset and tokenizer. 
Import IMDBDataset class from `./utils/train_utils`. 
Also join path from root directory to import from src withour errors about relative imports.
While implementing model training script you should log everything using logging library.
Also, log GPU memory usage using pynvml library.
For metrics use accuracy from evaluate module.
''', '''
{
    "code": """
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.arrow_dataset import Dataset
from fsspec.utils import tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, data
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
from typing import Any, Dict, List, Tuple
from datasets import DatasetDict, load_dataset
import logging
from pynvml import *
from utils.train_utils import IMDBDataset
torch.backends.cuda.matmul.allow_tf32 = True
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)

MODEL_STR_NAME = "bert-base-cased"
metric = evaluate.load("accuracy")

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."

def get_datasets():
    dataset: DatasetDict = load_dataset("imdb")
    logging.info(" Dataset loaded: {dataset}")
    return (dataset["train"], dataset["test"], dataset["unsupervised"])

def get_imdb_datasets(tokenizer: BertTokenizerFast):
    train_raw, val_raw, pred_raw = get_datasets()
    output = (IMDBDataset(tokenizer, ds) for ds in [train_raw, val_raw, pred_raw])
    logging.info(" IMDBDataset splits loaded ")
    return output

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_STR_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_STR_NAME)
model.to("cuda")

logging.info(f"Model Loaded: {print_gpu_utilization()}")
train_ds, val_ds, pred_ds = get_imdb_datasets(tokenizer)

train_args = TrainingArguments(
    output_dir="./model/checkpoints",
    evaluation_strategy="epoch",
    optim="adafactor",  # for performace purposes
    learning_rate=2e-5,
    max_grad_norm=0.3,
    num_train_epochs=9,
    # use_cpu=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    # bf16=True,
    tf32=True,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

logging.info("All set up, running model train... ")
trainer.train()
""",
    filepath: "src/model.py"
}
''')]

    class Analyst:
        system_message = """
As a business analyst, your task is to analyze the user feedback provided from the code running and translate it into a comprehensive requirements list. This list will serve as a foundation for future development efforts, ensuring that the software aligns with user expectations and addresses any issues or missing features.
Later you will be provided with user feedback and things that developer team already tried.
"""

        prompt = """
1. Review the user feedback carefully:
   - Identify specific pain points, challenges, or areas of confusion reported by users.
   - Note any missing features or functionalities requested by users.
   - Look for suggestions or ideas for improvements or enhancements.

2. Categorize the feedback:
   - Group similar feedback items together to identify common themes or patterns.
   - Prioritize the feedback based on its importance, impact on user experience, and alignment with business goals.

3. Translate the feedback into requirements:
   - For each category or theme, formulate clear and concise requirements that address the user feedback.
   - Use appropriate language and terminology that can be easily understood by both technical and non-technical stakeholders.
   - Ensure that the requirements are specific, measurable, achievable, relevant, and time-bound (SMART).

4. No yapping.
5. In your answer reffer as "you" to developer, that will implement everything.
User feedback:
"""
        shots = []


    class Namer:
        system_message = """You are an AI assistant tasked with naming code files based on their content. The file names should be concise, descriptive, and no longer than 15 characters."""
        prompt = """Given the code for a program, function, or module, provide a short but descriptive file name for it, adhering to the 15-character limit. The file name should be relevant to the code's functionality and purpose. Use only ASCII characters and avoid special characters or spaces. 
        file shuld start with `run_` and end with `.py`.
        """
        shots = [
            ("""```py
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a
```""", """run_fib_sequence.py"""),
            ("""function calculateTotal(items) {
  let total = 0;
  for (let item of items) {
    total += item.price * item.quantity;
  }
  return total;
}```""", 
"""run_calc_total.js"""),
        ]
    
    class Summary:
        system_message = """"""
        prompt = """"""
        shots = []    

    class Debugger:
        system_message = """debugger system_message"""
        prompt = """debugger prompt"""
        shots = []
    
    
    class Consoler:
        system_message = """ """
        prompt = """ """
        shots = [
            (""" """, """ """),
            (""" """, """ """),
        ]