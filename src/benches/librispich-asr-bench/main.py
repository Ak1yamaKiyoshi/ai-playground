import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from datasets import load_dataset

dataset = load_dataset('librispeech_asr')



#!
#!
#!
#!
#!



# research 
# whishper 3 
## https://huggingface.co/openai/whisper-large-v3/discussions/83
## https://github.com/egorsmkv/speech-recognition-uk?tab=readme-ov-file#whisper
#
## https://huggingface.co/blog/fine-tune-w2v2-bert