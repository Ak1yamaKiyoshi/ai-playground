import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))



from utils.train_utils import (CardiffTwitterSentimentDataset, compute_metrics)
from transformers import (AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from utils import print_gpu_utilization
from datasets import load_dataset
from peft import get_peft_model, PeftModel
from config.train_config.config import Config
import logging
import sys
import os
import warnings 
from transformers import logging as tlogging

tlogging.set_verbosity_error()
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

label2id = {
    "arts_&_culture": 0,
    "business_&_entrepreneurs": 1,
    "pop_culture": 2,
    "daily_life": 3,
    "sports_&_gaming": 4,
    "science_&_technology": 5
}

def construct_details(lora, optimizer, batchsize):
    return f"bench-1-{'lora-' if lora else ''}{optimizer.replace('_','-')}-batch-{batchsize}"

dataset_name = "cardiffnlp/tweet_topic_single"
model_name = "bert-base-cased"
def train(params, use_lora):
    train_with_lora = use_lora

    output_dict = {
        "model_name":model_name,
        "dataset_name":dataset_name,
        "details": construct_details(train_with_lora, params['optim'], params['per_device_train_batch_size'])
    }

    output_dir = Config.output_dir(**output_dict)
    log_dir = Config.log_dir(**output_dict)
    adapter_name = Config.adapter_name(**output_dict)

    logging.info(f"Output dir: {output_dir}\nINFO: Logging_dir {log_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, trust_remote_code=True)

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
    
    logging.info(f"\033[92m{output_dir.split('/')[-1]}\033[0m")
    logging.info(f"\033[94m{params}\033[0m")
    
    train_ds, val_ds = get_huggingface_splitted_datasets(
        CardiffTwitterSentimentDataset,tokenizer, dataset_name, label2id)

    train_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        evaluation_strategy="epoch",
        **params
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    params_path = log_dir + "/training-params-custom.txt"
    os.makedirs(log_dir, exist_ok=True)
    with open(params_path, "w+") as f:
        f.write(str(params))
    logging.info(f"Training args saved at: {params_path}")
    
    usage = print_gpu_utilization("Model Loaded")
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/memory_usage.txt", "w+") as f:
        f.write(usage)
    trainer.train()

trained_with_grad_steps_2_and_grad_norm_03 = [
    "paged_adamw_32bit", 
    "adamw_torch_fused",    
    "adamw_hf",
]
optimizers_to_test = [
    "paged_adamw_32bit", 
    "adamw_torch_fused",    
    "adamw_hf",
    "adafactor", # no data in bench 1
    "adamw_bnb_8bit", # no data in bench 1 
]
batch_sizes = [32, 16, 8 ]
grad_accumulation_steps = [1, 2,]
grad_norms = [0.1, 0.2, 0.3]

i = 0
while True:
    for optimizer in optimizers_to_test:
        for batch_size in batch_sizes:
            for gradaccumsteps in grad_accumulation_steps:
                if gradaccumsteps == 2 and optimizer in trained_with_grad_steps_2_and_grad_norm_03:
                    continue
                for grad_norm in grad_norms:
                    if grad_norms == 0.3 and optimizer in trained_with_grad_steps_2_and_grad_norm_03:
                        continue
                    i+=1 
                    params = {
                        "num_train_epochs":5,
                        "logging_steps":10,
                        "optim": optimizer,
                        "group_by_length":True,
                        "learning_rate":2e-5,
                        "max_grad_norm":grad_norms,
                        "per_device_train_batch_size":batch_size,
                        "per_device_eval_batch_size":1,
                        "gradient_accumulation_steps":gradaccumsteps,
                        "gradient_checkpointing":True,
                        "fp16":True,
                        "tf32":True,
                        "metric_for_best_model":"f1",
                    }
                    try:
                        train(params, False)
                    except: 
                        logging.info("Skipped iteration")
                        continue
                    logging.info(i)