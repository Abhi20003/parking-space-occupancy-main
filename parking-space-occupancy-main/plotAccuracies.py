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
import csv

from datasets import acpds
from utils import transforms
from utils import visualize as vis

weights = ["parking-space-occupancy-main/pd-camera-weights/RCNN_ResNet50_128_square_0/train_log.csv",
           "parking-space-occupancy-main/pd-camera-weights/RCNN_MobileNetv2_128_square_0/train_log.csv",
           "parking-space-occupancy-main/pd-camera-weights/FasterRCNN_FPN_MobileNetv3_1440_square_0/train_log.csv",
           "parking-space-occupancy-main/pd-camera-weights/FasterRCNN_FPN_ResNet50_1440_square_0/train_log.csv"]

model_names = ["RCNN_ResNet50 (roi_res = 128)", "RCNN_MobileNetv2 (roi_res = 128)", "FasterRCNN_FPN_MobileNetv2 (roi_res = 1440)", "FasterRCNN_FPN_ResNet50 (roi_res = 1440)"]

for i in range(4):
    epochs = [i for i in range(1,101)]
    train_accuracy = []
    valid_accuracy = []
    cnt=0
    sum_train=0
    sum_val=0
    with open(weights[i], mode ='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            cnt+=1
            if cnt==1: continue
            train_accuracy.append(float(lines[1])*100)
            valid_accuracy.append(float(lines[3])*100)
            sum_train += float(lines[1])*100
            sum_val += float(lines[3])*100
            # print(lines[1])
    print(model_names[i])
    print("Training Accuracy Average", (sum_train/(cnt-1)))
    print("Test Accuracy Average", sum_val/(cnt-1))
    plt.figure(i)
    plt.plot(epochs, train_accuracy, label="Training Accuracy")
    plt.plot(epochs, valid_accuracy, label="Test Accuracy")
    # plt.title(model_names[i])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(model_names[i]+"_accuracy.png")