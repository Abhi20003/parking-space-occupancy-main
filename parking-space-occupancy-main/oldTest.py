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


# # create model
# modelsRCNN = [RCNN(roi_res=64), RCNN(roi_res=128), RCNN(roi_res=192), RCNN(roi_res=256)]
# modelsFRCNN = [FasterRCNN_FPN(roi_res=5), FasterRCNN_FPN(roi_res=7), FasterRCNN_FPN(roi_res=10)]
model = RCNN(roi_res=128, pooling_type='qdrl')


print("dowloding dataset")

# if not os.path.exists('dataset/data'):
#     r = requests.get("https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip")
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall('dataset/data')
    
print("donee dataset")

train_ds, valid_ds, test_ds = acpds.create_datasets('dataset/data')


# load model weights
weights_path = 'weights.pt'
# if not os.path.exists(weights_path):
#     r = requests.get('https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2FRCNN_128_square_gopro.pt')
#     with open(weights_path, 'wb') as f:
#         f.write(r.content)



# # Plot Test Predictions


# print("Started")


# lsRCNN = ["R-CNN (64)", "R-CNN (128)","R-CNN (192)", "R-CNN (256)"]

# XRCNN = [64,128,192,256]
# XFRCNN = [5,7,10]

# YRCNN = [0]*len(modelsRCNN)
# YFRCNN = [0]*len(modelsFRCNN)



model.load_state_dict(torch.load(weights_path, map_location='cpu'))
for i, (image_batch, rois_batch, labels_batch) in enumerate(test_ds):
    if i==5: break
    image, rois = image_batch[0], rois_batch[0]
    image = transforms.preprocess(image)
    with torch.no_grad():
        class_logits = model(image, rois)
        class_scores = class_logits.softmax(1)[:, 1]
    L = vis.plot_ds_image(image, rois, class_scores)
    # print(i, L[0], L[1])
    # ele = [L[1], L[0]]
    # mod_arr.insert(0,ele)
# mod_arr.sort(key = lambda x: x[0])
# Xarr, Yarr = np.array(mod_arr).T
# print(lsRCNN[j], np.average(Yarr), "done")
# YRCNN[j] = np.average(Yarr)
# plt.figure(1)
# plt.plot(Xarr, Yarr, label=lsRCNN[j])

# plt.title('RCNN model')
# plt.xlabel('Number of Parking Spaces')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig("1.png")

# plt.figure(2)
# plt.plot(XRCNN, YRCNN)
# plt.title('RCNN model')
# plt.xlabel('Roi_Res value')
# plt.ylabel('Average Accuracy')
# plt.savefig("2.png")


# for j in range(0, len(modelsFRCNN)):
#     mod_arr = []
#     model = modelsFRCNN[j]
#     for i, (image_batch, rois_batch, labels_batch) in enumerate(test_ds):
#         image, rois, labels = image_batch[0], rois_batch[0], labels_batch[0]
#         image = transforms.preprocess(image)
#         with torch.no_grad():
#             class_logits = model(image, rois)
#             class_scores = class_logits.softmax(1)[:, 1]
#         L = vis.plot_ds_image(image, rois, class_scores, labels)
#         print(i, L[0], L[1])
#         ele = [L[1], L[0]]
#         mod_arr.insert(0,ele)
#     mod_arr.sort(key = lambda x: x[0])
#     Xarr, Yarr = np.array(mod_arr).T
#     print(lsFRCNN[j], np.average(Yarr), "done")
#     YFRCNN[j] = np.average(Yarr)
#     plt.figure(3)
#     plt.plot(Xarr, Yarr, label=lsFRCNN[j])


# plt.title('FasterRCNN_FPN model')
# plt.xlabel('Number of Parking Spaces')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig("3.png")

# plt.figure(4)
# plt.plot(XFRCNN, YFRCNN)
# plt.title('FasterRCNN_FPN model')
# plt.xlabel('Roi_Res value')
# plt.ylabel('Average Accuracy')
# plt.savefig("4.png")

