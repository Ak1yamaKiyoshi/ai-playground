

class Config:
    log_dir = "./logs/"
    pricing = {
        'gpt-3.5-turbo-1106': {'input_price': 0.0010, 'output_price': 0.0020},
        'gpt-3.5-turbo-0613': {'input_price': 0.0015, 'output_price': 0.0020},
        'gpt-3.5-turbo-16k-0613': {'input_price': 0.0030, 'output_price': 0.0040},
        'gpt-3.5-turbo-0301': {'input_price': 0.0015, 'output_price': 0.0020},
        'gpt-3.5-turbo-0125': {'input_price': 0.0005, 'output_price': 0.0015},
        'gpt-3.5-turbo-instruct': {'input_price': 0.0015, 'output_price': 0.002},
        'gpt-4': {'input_price': 0.03, 'output_price': 0.06},
        'gpt-4-0125-preview': {'input_price': 0.01, 'output_price': 0.03},
        'gpt-4-1106-preview': {'input_price': 0.01, 'output_price': 0.03},
        'gpt-4-1106-vision-preview': {'input_price': 0.01, 'output_price': 0.03},
        'gpt-4-32k': {'input_price': 0.06, 'output_price': 0.12}
    }