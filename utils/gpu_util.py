from pynvml import *
import logging

def print_gpu_utilization(text:str="something"):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logging.info(f"GPU memory occupied after {text}: {info.used//1024**2} MB.")
