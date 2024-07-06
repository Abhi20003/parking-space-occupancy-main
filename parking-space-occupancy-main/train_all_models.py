import os
import io
import json
import requests
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from collections import namedtuple
from glob import glob


# download the training logs
logs_dir = 'training_output'
if not os.path.exists(logs_dir):
    r = requests.get("https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Fpaper_training_output.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(logs_dir)
    
    
# # create dicts with model validation and test accuracies
# va_dict = defaultdict(list)
# ta_dict = defaultdict(list)

# # iterate through model directories
# for model_dir in sorted(glob(f'{logs_dir}/*')):
    
#     # get model id based on model directory
#     model_id = model_dir.split('/')[-1]
    
#     # split model_id into model_name and training_iter
#     model_name, _ = model_id.rsplit('_', 1)
    
#     # read validation accuracy from training logs 
#     train_log = pd.read_csv(f'{model_dir}/train_log.csv')
#     va = train_log.valid_accuracy.tolist()
    
#     # append logs if they're the first logs of the given model
#     # or if they're of the same length as the previous logs
#     # (avoid storing logs of a model that hasn't finished trainig yet) 
#     if len(va_dict[model_name]) == 0 or len(va_dict[model_name][0]) == len(va):
#         # read test accuracy from test logs
#         with open(f'{model_dir}/test_logs.json') as f:
#             ta = json.load(f)['accuracy']
            
#         va_dict[model_name] += [va]
#         ta_dict[model_name] += [ta]

# # compute accuracy mean and SE for each model
# Logs = namedtuple('Logs', ['va_mean', 'va_se', 'ta_mean', 'ta_se'])
# logs = {}
# for k, v in va_dict.items():
#     # print number of training iters for each model
#     print(f'{k}: {len(v)}')

#     # calculate the mean and standard error of valid. accuracy
#     va = np.array(v)
#     # va = np.array([ma(x, 10) for x in va])
#     va_mean = np.mean(va, 0)
#     va_se = np.std(va, 0) / np.sqrt(va.shape[0])
    
#     # calculate the mean and standard error of test accuracy
#     ta = np.array(ta_dict[k])
#     ta_mean = np.mean(ta)
#     ta_se = np.std(ta) / np.sqrt(len(ta))
    
#     # save validation and test logs
#     logs[k] = Logs(va_mean, va_se, ta_mean, ta_se)
    
# def ma(x, w=10):
#     """Moving average."""
#     return np.convolve(x, np.ones(w), 'valid') / w

# fig, ax = plt.subplots(figsize=[12, 8])
# for k, v in logs.items():
#     epochs = np.arange(len(v.va_mean))
#     plt.plot(epochs, v.va_mean, label=k, linewidth=2)
    
# plt.xlabel('Epochs')
# plt.ylabel('Average Accuracy')
# ax.legend()
# ax.set_ylim([0.925, 0.99])
# plt.savefig("Accuracy vs epochs.png")