from utils import tflog2pandas, find_files
import glob
import shutil


path_to_logs = './output/logs'
fucked_up_logs_dir = "./temp-logdir/"
logs = find_files(path_to_logs, "events.out")

for log in logs:
    df = tflog2pandas(log)
    print(df.sample(5))
    #! Load params file from these directories .split("/")[:-1]
    #! and merge them somehow 

""" 
find_files(directory, prefix)

path=""
df=tflog2pandas(path)
#df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
df.to_csv("output.csv")
"""