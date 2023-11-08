import os
import io
import time
import requests
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from dataset import acpds
from utils import transforms
from utils import visualize as vis

print("dowloding dataset")

# if not os.path.exists('dataset/data'):
#     r = requests.get("https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip")
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall('dataset/data')
    
print("donee dataset")

train_ds, valid_ds, test_ds = acpds.create_datasets('dataset/data')

# Visualize data

# image_batch, rois_batch, labels_batch = next(iter(valid_ds))
# new_iter = iter(valid_ds)
# image_batch1, rois_batch1, labels_batch1 = next(new_iter)
# image_batch2, rois_batch2, labels_batch2= next(new_iter)
# image_raw, rois, labels = image_batch[0], rois_batch[0], labels_batch[0]
# image_raw1, rois1, labels1 = image_batch1[0], rois_batch1[0], labels_batch1[0]
# image_raw2, rois2, labels2 = image_batch2[0], rois_batch2[0], labels_batch2[0]
# image = transforms.preprocess(image_raw, res=1440)
# image1 = transforms.preprocess(image_raw1, res=1440)
# image2 = transforms.preprocess(image_raw2, res=1440)
# vis.plot_ds_image(image, rois, labels, show=True)
# # vis.plot_ds_image(image1, rois1, labels, show=True)
# # vis.plot_ds_image(image2, rois2, labels, show=True)
# print("Done")


# image_raw, rois, labels = image_batch[0], rois_batch[0], labels_batch[0]
# image_aug, rois_aug = transforms.augment(image_raw, rois)
# image_aug = transforms.preprocess(image_aug, res=1440)
# vis.plot_ds_image(image_aug, rois_aug, labels, show=True)   

# create model
from models.rcnn import RCNN
model = RCNN()

# load model weights
weights_path = 'weights.pt'
if not os.path.exists(weights_path):
    r = requests.get('https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2FRCNN_128_square_gopro.pt')
    with open(weights_path, 'wb') as f:
        f.write(r.content)
model.load_state_dict(torch.load(weights_path, map_location='cpu'))

# Plot Test Predictions


for i, (image_batch, rois_batch, labels_batch) in enumerate(test_ds):
    if i == 2: break
    image, rois, labels = image_batch[0], rois_batch[0], labels_batch[0]
    image = transforms.preprocess(image)
    with torch.no_grad():
        class_logits = model(image, rois)
        class_scores = class_logits.softmax(1)[:, 1]
    vis.plot_ds_image(image, rois, class_scores)
    
    
#Speed comparison
    
from models.rcnn import RCNN
from models.faster_rcnn_fpn import FasterRCNN_FPN

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

times['R-CNN (256)'] = time_model(RCNN(roi_res=256), res=[4000, 3000])
print("times 256 defined")

times['Faster R-CNN FPN (800)']   = time_model(FasterRCNN_FPN(), res=[1067, 800])
print("times 800 defined")

times['Faster R-CNN FPN (1100)']  = time_model(FasterRCNN_FPN(), res=[1467, 1100])
print("times 1100 defined")

times['Faster R-CNN FPN (1440)']  = time_model(FasterRCNN_FPN(), res=[1920, 1440])
print("times 1440 defined")

print("times defined")

for k, v in times.items():
    plt.plot(v[0], v[1], label=k)
    plt.fill_between(v[0], v[1]-v[2], v[1   ]+v[2], alpha=0.5)
    
print("for loop executed")

plt.title('Inference Time')
plt.xlabel('Number of Parking Spaces')
plt.ylabel('Time [Seconds]')
plt.legend()
plt.show()