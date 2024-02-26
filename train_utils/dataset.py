
from typing import Dict, List
import torch
from datasets import DatasetDict, load_dataset
import logging


def invert_dict(to_invert_:Dict) -> Dict:
    return {v: k for k, v in to_invert_.items()}

def get_huggingface_splitted_datasets(cls, tokenizer, name):
    try:
        dataset = load_dataset(name)
        train_raw, val_raw, pred_raw = (dataset["train"], dataset["test"], dataset["unsupervised"])
        output = (cls(tokenizer, ds) for ds in [train_raw, val_raw, pred_raw])
    except Exception as e:
        logging.error(f"{e}")
    logging.info(f"{name[0].upper()}{name[0:]} splits loaded ")
    return output


class Dataset:
    def __init__(self,
            iterable_dataset,
            tokenizer, 
            max_len:int, 
        ) -> None:
        self.tokenizer = tokenizer
        self.iterable_dataset = iterable_dataset
        self.max_len = max_len

    def __getitem__(self, index) -> Dict[str]:
        label = self.iterable_dataset[index]["label"]
        text = self.iterable_dataset[index]["text"]
        outputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len,
        )
        return {
            "input_ids": outputs["input_ids"].squeeze(0),
            "attention_mask": outputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.int8),
        }
    
    def __len__(self):
        return len(self.iterable_dataset)


class IMDBDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        super().__init__(dataset, tokenizer, 512)


class CardiffTwitterSentimentDataset(Dataset):
    def __init__(self, iterable_dataset, tokenizer, label2id) -> None:
        super().__init__(iterable_dataset, tokenizer, 512)
        self.label2id = label2id
        self.id2label = invert_dict(self.label2id)