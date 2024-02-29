import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, purge_orphaned_data=False)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def find_files(directory, prefix):
    files__ = []
    j = 0
    for root, _, files in os.walk(directory):
        if _ != []:
            dirnames = _

        for i, filename in enumerate(files):
            if filename == []:
                continue
            if filename.startswith(prefix):
                file__ = '/'.join([
                    directory,
                    dirnames[j],
                    filename
                ])
                
                files__.append(file__)
        if files != []:
            j+=1
    return files__