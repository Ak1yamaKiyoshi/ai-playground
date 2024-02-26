from datetime import datetime

class Config:

    @classmethod
    def output_dir(model_name:str, details:str, dataset_name:str, directory:str="./checkpoints/") -> str:
        dataset = dataset_name.split("/")[-1]
        model = model_name.split("/")[-1]
        current_time = datetime.now().strftime('%H-%M')
        return f"{directory}{dataset}.{model}.{current_time}"