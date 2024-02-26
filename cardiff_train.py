from train_utils import (CardiffTwitterSentimentDataset, get_huggingface_splitted_datasets, compute_metrics)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from utils import print_gpu_utilization
import evaluate
import logging


metric = evaluate.load("accuracy")
dataset_name = "cardiffnlp/tweet_topic_single"
model_name = "bert-base-cased"
output_dir = f""

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to("cuda")

print_gpu_utilization("Model Loading")

train_ds, val_ds, pred_ds = get_huggingface_splitted_datasets(
    CardiffTwitterSentimentDataset,tokenizer, dataset_name)

train_args = TrainingArguments(
    output_dir="./model/checkpoints",
    evaluation_strategy="epoch",
    optim="adafactor", 
    learning_rate=2e-5,
    max_grad_norm=0.3,
    num_train_epochs=9,
    # use_cpu=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True, 
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

# https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
