from peft import LoraConfig, TaskType
from datetime import datetime

class Config:
    @staticmethod
    def log_dir(**params):
        params["directory"] = params.get("directory", "./output/logs/")
        return Config.output_dir(**params)
    
    @staticmethod
    def output_dir(**params) -> str:
        """ 
        model_name:str, details:str, dataset_name:str, directory:str="./checkpoints/"
        """
        dataset = params.get('dataset_name', "custom").split("/")[-1]
        model = params.get("model_name", "unknown-model").split("/")[-1]
        directory = params.get("directory", "./output/checkpoints/")
        details = params.get("details", "none")
        current_time = datetime.now().strftime('(%Y-%m-%d)-(%H:%M:%S)')
        return f"{directory}{dataset}-{model}-{details}-{current_time}"
    
    @staticmethod
    def lora():
        return LoraConfig(
            r=16,
            lora_alpha=1,
            target_modules=["query", "value"],
            lora_dropout=0.01,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )

    @staticmethod
    def adapter_name(**params):
        params['directory'] = ""
        params['details'] = params.get('details', "") + "peft" 
        return Config.output_dir(**params)