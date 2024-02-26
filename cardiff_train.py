from train_utils import (CardiffTwitterSentimentDataset, compute_metrics)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from utils import print_gpu_utilization
from datasets import load_dataset
from train_config.config import Config
import evaluate
import logging
import sys
import os

def get_huggingface_splitted_datasets(cls, tokenizer, dataset_name, label2id):
    dataset_dict = load_dataset(dataset_name)
    """ 
    ['test_2020', 'test_2021', 'train_2020', 'train_2021', 'train_all',
    'validation_2020', 'validation_2021', 'train_random', 'validation_random', 
    'test_coling2022_random', 'train_coling2022_random', 'test_coling2022', 'train_coling2022']
    """
    return (cls(dataset_dict['train_all'], tokenizer, label2id),
            cls(dataset_dict['validation_random'], tokenizer, label2id)
    )

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)

label2id = {
    "arts_&_culture": 0,
    "business_&_entrepreneurs": 1,
    "pop_culture": 2,
    "daily_life": 3,
    "sports_&_gaming": 4,
    "science_&_technology": 5
}

metric = evaluate.load("accuracy")
dataset_name = "cardiffnlp/tweet_topic_single"
model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
model.to("cuda")

train_ds, val_ds = get_huggingface_splitted_datasets(
    CardiffTwitterSentimentDataset,tokenizer, dataset_name, label2id)

print_gpu_utilization("Model Loading")

train_args = TrainingArguments(
    output_dir=Config.output_dir(model_name, "", dataset_name),
    logging_dir="./logs"+Config.output_dir(model_name, "", dataset_name),
    logging_steps=10,
    evaluation_strategy="epoch",
    optim="adafactor", 
    group_by_length=True,
    learning_rate=2e-5,
    max_grad_norm=0.3,
    num_train_epochs=45,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True, 
    tf32=True,
    # use_cpu=True,
)

train_args_dict = vars(train_args)
train_args_filepath = Config.output_dir(model_name, "", dataset_name)+"/stats/train_args.json"
os.makedirs('/'.join(train_args_filepath.split("/")[:-1]), exist_ok=True)
with open(train_args_filepath, "w") as f:
    f.write(str(train_args_dict))

logging.info(f"Training args dumped as dict at: {train_args_filepath}")

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

logging.info("All set up, running model train... ")
trainer.train()

# https://huggingface.co/datasets/cardiffnlp/tweet_topic_single