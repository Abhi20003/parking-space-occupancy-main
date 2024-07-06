import os
import io
import time
import requests
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from datasets import acpds
from utils import transforms
from utils import visualize as vis

from models.rcnn import RCNN
from models.faster_rcnn_fpn import FasterRCNN_FPN
from utils.engine import train_model
from models.rcnn import RCNN
from models.faster_rcnn_fpn import FasterRCNN_FPN

from collections import defaultdict
from collections import namedtuple
from glob import glob


print("dowloding dataset")

# if not os.path.exists('dataset1/data'):
#     r = requests.get("https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip")
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall('dataset/data')
    
print("donee dataset")

train_ds, valid_ds, test_ds = acpds.create_datasets('dataset1/data')


# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set dir to store model weights and logs
wd = "training_output"

print("done")

# train each model multiple times
for i in range(5):
    # RCNN
    # train_model(RCNN(roi_res=64,  pooling_type='qdrl'),   train_ds, valid_ds, test_ds, f'{wd}/RCNN_64_qdrl_{i}',    device)
    # train_model(RCNN(roi_res=128, pooling_type='qdrl'),   train_ds, valid_ds, test_ds, f'{wd}/RCNN_128_qdrl_{i}',   device)
    # train_model(RCNN(roi_res=256, pooling_type='qdrl'),   train_ds, valid_ds, test_ds, f'{wd}/RCNN_256_qdrl_{i}',   device)
    train_model(RCNN(roi_res=64,  pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/RCNN_64_square_{i}',  device)
    train_model(RCNN(roi_res=128, pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/RCNN_128_square_{i}', device)
    train_model(RCNN(roi_res=256, pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/RCNN_256_square_{i}', device)

    # FasterRCNN_FPN
    # train_model(FasterRCNN_FPN(pooling_type='qdrl'),   train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_1440_qdrl_{i}',   device, res=1440)
    train_model(FasterRCNN_FPN(pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_1440_square_{i}', device, res=1440)
    # train_model(FasterRCNN_FPN(pooling_type='qdrl'),   train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_1100_qdrl_{i}',   device, res=1100)
    train_model(FasterRCNN_FPN(pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_1100_square_{i}', device, res=1100)
    # train_model(FasterRCNN_FPN(pooling_type='qdrl'),   train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_800_qdrl_{i}',    device, res=800)
    train_model(FasterRCNN_FPN(pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_800_square_{i}',  device, res=800)

# download the training logs
logs_dir = wd
    
# create dicts with model validation and test accuracies
va_dict = defaultdict(list)
ta_dict = defaultdict(list)

# iterate through model directories
for model_dir in sorted(glob(f'{logs_dir}/*')):
    
    # get model id based on model directory
    model_id = model_dir.split('/')[-1]
    
    # split model_id into model_name and training_iter
    model_name, _ = model_id.rsplit('_', 1)
    
    # read validation accuracy from training logs 
    train_log = pd.read_csv(f'{model_dir}/train_log.csv')
    va = train_log.valid_accuracy.tolist()
    
    # append logs if they're the first logs of the given model
    # or if they're of the same length as the previous logs
    # (avoid storing logs of a model that hasn't finished trainig yet) 
    if len(va_dict[model_name]) == 0 or len(va_dict[model_name][0]) == len(va):
        # read test accuracy from test logs
        with open(f'{model_dir}/test_logs.json') as f:
            ta = json.load(f)['accuracy']
            
        va_dict[model_name] += [va]
        ta_dict[model_name] += [ta]

# compute accuracy mean and SE for each model
Logs = namedtuple('Logs', ['va_mean', 'va_se', 'ta_mean', 'ta_se'])
logs = {}
for k, v in va_dict.items():
    # print number of training iters for each model
    print(f'{k}: {len(v)}')

    # calculate the mean and standard error of valid. accuracy
    va = np.array(v)
    # va = np.array([ma(x, 10) for x in va])
    va_mean = np.mean(va, 0)
    va_se = np.std(va, 0) / np.sqrt(va.shape[0])
    
    # calculate the mean and standard error of test accuracy
    ta = np.array(ta_dict[k])
    ta_mean = np.mean(ta)
    ta_se = np.std(ta) / np.sqrt(len(ta))
    
    # save validation and test logs
    logs[k] = Logs(va_mean, va_se, ta_mean, ta_se)
    
def ma(x, w=10):
    """Moving average."""
    return np.convolve(x, np.ones(w), 'valid') / w

plt.figure(0)
fig, ax = plt.subplots(figsize=[12, 8])
for k, v in logs.items():
    epochs = np.arange(len(v.va_mean))
    plt.figure(0)
    plt.plot(epochs, v.va_mean, label=k, linewidth=2)
    
plt.xlabel('Epochs')
plt.ylabel('Average Accuracy')
ax.legend()
ax.set_ylim([0.925, 0.99])
plt.savefig("Accuracy vs epochs.png")
    
#Speed comparison
    


def time_model(model, res=[1920, 1440], n=3):
    image = torch.zeros([3, res[1], res[0]])
    L_arr = torch.linspace(1, 100, steps=10, dtype=torch.int32)
    mean = np.zeros_like(L_arr, dtype=np.float32)
    ese = np.zeros_like(L_arr, dtype=np.float32)
    for i, L in enumerate(L_arr):
        times = np.zeros(n)
        for j in range(n):
            rois = torch.rand([L, 4, 2])
            t0 = time.time()
            out = model(image, rois)
            t1 = time.time()
            times[j] = t1 - t0
        mean[i] = np.mean(times)
        ese[i] = np.std(times) / np.sqrt(len(times))
    return L_arr, mean, ese

print("Function Defined")

times = {}
times['R-CNN (64)']  = time_model(RCNN(roi_res=64),  res=[4000, 3000])
print("times 64 defined")

times['R-CNN (128)'] = time_model(RCNN(roi_res=128), res=[4000, 3000])
print("times 128 defined") 

times['R-CNN (192)']  = time_model(RCNN(roi_res=192),  res=[4000, 3000])
print("times 192 defined")

times['R-CNN (256)'] = time_model(RCNN(roi_res=256), res=[4000, 3000])
print("times 256 defined")

times['Faster R-CNN FPN (800)'] = time_model(FasterRCNN_FPN(), res=[1067, 800])
print("times 800 defined")

times['Faster R-CNN FPN (1100)'] = time_model(FasterRCNN_FPN(), res=[1467, 1100])
print("times 1100 defined")

times['Faster R-CNN FPN (1440)'] = time_model(FasterRCNN_FPN(), res=[1920, 1440])
print("times 1440 defined")


print("times defined")

for k, v in times.items():
    plt.figure(1)
    plt.plot(v[0], v[1], label=k)
    plt.fill_between(v[0], v[1]-v[2], v[1]+v[2], alpha=0.5)
    
print("for loop executed")

plt.title('Inference Time')
plt.xlabel('Number of Parking Spaces')
plt.ylabel('Time [Seconds]')
plt.legend()
plt.savefig("Inference_Time.png")