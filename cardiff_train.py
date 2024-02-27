from train_utils import (CardiffTwitterSentimentDataset, compute_metrics)
from transformers import (AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from utils import print_gpu_utilization
from datasets import load_dataset
from peft import get_peft_model, PeftModel
from train_config.config import Config
import logging
import sys
import os
import safetensors

def get_test_datasets(cls, tokenizer, dataset_name, label2id):
    dataset_dict = load_dataset(dataset_name)
    final_dataset = []
    for dataset in (dataset_dict[name] for name in ["test_coling2022", "test_2020", "test_2021"]):
        for i in range(dataset):
            final_dataset.append(i)
    return final_dataset


def get_huggingface_splitted_datasets(cls, tokenizer, dataset_name, label2id):
    dataset_dict = load_dataset(dataset_name)
    """ 
        ['test_2020', 'test_2021', 'train_2020', 'train_2021', 'train_all',
        'validation_2020', 'validation_2021', 'train_random', 'validation_random', 
        'test_coling2022_random', 'train_coling2022_random', 'test_coling2022', 'train_coling2022']
    """
    return (cls(dataset_dict['train_all'], tokenizer, label2id),
            cls(dataset_dict['test_coling2022'], tokenizer, label2id)
    )


logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)


dataset_name = "cardiffnlp/tweet_topic_single"
model_name = "bert-base-cased"

train_with_lora = False
load_from_checkpoint = False
checkpoint_name = "./output/checkpoints/tweet_topic_single-bert-base-cased-base-(2024-02-27)-(14:21:32)/checkpoint-1000/model.safetensors"
checkpoint_model_config = "./output/checkpoints/tweet_topic_single-bert-base-cased-base-(2024-02-27)-(14:21:32)/checkpoint-1000/config.json"

params = {
    "num_train_epochs":15,
    "logging_steps":10,
    "optim": "adamw_bnb_8bit", #"paged_adamw_32bit",#"adamw_torch_fused", #"adagrad", #"adamw_hf", #"adamw_torch",
    "group_by_length":True,
    "learning_rate":2e-5,
    "max_grad_norm":0.3,
    "per_device_train_batch_size":4,
    "per_device_eval_batch_size":1,
    "gradient_accumulation_steps":2,
    "gradient_checkpointing":True,
    "fp16":True,
    "tf32":True,
    "metric_for_best_model":"f1",
    "weight_decay": 1e-5,
}

output_dict = {
    "model_name":model_name,
    "dataset_name":dataset_name,
    "details":f"{'lora' if train_with_lora else 'base'}-{params['optim']}",
}

output_dir = Config.output_dir(**output_dict)
log_dir = Config.log_dir(**output_dict)
adapter_name = Config.adapter_name(**output_dict)

label2id = {
    "arts_&_culture": 0,
    "business_&_entrepreneurs": 1,
    "pop_culture": 2,
    "daily_life": 3,
    "sports_&_gaming": 4,
    "science_&_technology": 5
}

logging.info(f"Output dir: {output_dir}\n Logging_dir {log_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

if load_from_checkpoint:
    if train_with_lora: raise Warning("Training from checkpoint with LoRA often degrades the result")
    safetensors.torch.load_model(model, checkpoint_name)

if train_with_lora:
    peft_model = get_peft_model(model, Config.lora(), adapter_name)
    peft_model.print_trainable_parameters()
    peft_model.to("cuda")

    for name, param in peft_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            logging.info(f"Unfrozen: {name}")
else:
    model.to("cuda")
 
train_ds, val_ds = get_huggingface_splitted_datasets(
    CardiffTwitterSentimentDataset,tokenizer, dataset_name, label2id)

print_gpu_utilization("Model Loading")

train_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=log_dir,
    report_to=["tensorboard"],
    evaluation_strategy="epoch",
    **params
)

params_path = output_dir + "/training-params-custom.txt"
os.makedirs(output_dir, exist_ok=True)
with open(params_path, "w") as f:
    f.write(str(params))
logging.info(f"Training args saved at: {params_path}")

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

logging.info("All set up, running model train... ")
trainer.train()



# Some of this is used, some of it is not
# Dataset
# https://huggingface.co/datasets/cardiffnlp/tweet_topic_single
# Hyperparameter search reference: 
# https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
# PEFT lora
# https://jaotheboss.medium.com/peft-with-bert-8763d8b8a4ca
# LORA more detailed guide 
# https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
# When to use LoRA 
# https://crunchingthedata.com/when-to-use-lora/#:~:text=LoRA%20is%20a%20model%20fine,parameters%20that%20are%20not%20frozen.
# Kfold cross validation
# https://stackoverflow.com/questions/75510487/huggingface-trainer-k-fold-cross-validation

# Todo: 
# If LoRA, save checkpoint name to config
# Implement RLHF    
# test RoBERTa + DeBERTa