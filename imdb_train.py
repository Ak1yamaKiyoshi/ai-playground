#!pip install evaluate -q
from datasets.arrow_dataset import Dataset
from fsspec.utils import tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, data
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import os
import sys
from typing import Any, Dict, List, Tuple
from datasets import DatasetDict, load_dataset
import logging
from pynvml import *
from train_utils import IMDBDataset
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
    fp16=True,  #! Oh god
    # bf16=True,#! It looks
    tf32=True,  #! Kinda scary
)

#! Todo: use deepspeed
#! Todo: log predictions and eval to tensorboard
#! Todo: upload model to huggingface
#! todo: Population Based Training

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

logging.info("All set up, running model train... ")
trainer.train()