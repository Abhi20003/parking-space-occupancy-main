import os
import io
import time
import requests
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt

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

image_batch, rois_batch, labels_batch = next(iter(valid_ds))
new_iter = iter(valid_ds)
image_batch1, rois_batch1, labels_batch1 = next(new_iter)
image_batch2, rois_batch2, labels_batch2= next(new_iter)
image_raw, rois, labels = image_batch[0], rois_batch[0], labels_batch[0]
image_raw1, rois1, labels1 = image_batch1[0], rois_batch1[0], labels_batch1[0]
image_raw2, rois2, labels2 = image_batch2[0], rois_batch2[0], labels_batch2[0]
image = transforms.preprocess(image_raw, res=1440)
image1 = transforms.preprocess(image_raw1, res=1440)
image2 = transforms.preprocess(image_raw2, res=1440)
vis.plot_ds_image(image, rois, labels, show=True)
vis.plot_ds_image(image1, rois1, labels, show=True)
vis.plot_ds_image(image2, rois2, labels, show=True)
print("Done")