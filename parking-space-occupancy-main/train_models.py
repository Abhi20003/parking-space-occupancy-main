import os
import torch

from datasets import acpds
from utils.engine import train_model
from models.rcnn import RCNN_ResNet50, RCNN_MobileNetv2
from models.rcnn_gcn import RCNN_GCN_ResNet50, RCNN_GCN_MobileNetv3
from models.faster_rcnn_fpn import FasterRCNN_FPN_ResNet50, FasterRCNN_FPN_MobileNetv2


# set device
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print(device)

# load dataset
train_ds, valid_ds, test_ds = acpds.create_datasets('dataset/data')

# set dir to store model weights and logs
wd = os.path.expanduser('~/Downloads/pd-camera-weights/')

# train each model multiple times
for i in range(1):
    # RCNN
    train_model(RCNN_ResNet50(roi_res=128,  pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/RCNN_ResNet50_128_square_{i}',  device)
    train_model(RCNN_MobileNetv2(roi_res=128, pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/RCNN_MobileNetv2_128_square_{i}', device)

    # FasterRCNN_FPN
    train_model(FasterRCNN_FPN_ResNet50(pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_ResNet50_1440_square_{i}', device, res=1440)
    train_model(FasterRCNN_FPN_MobileNetv2(pooling_type='square'), train_ds, valid_ds, test_ds, f'{wd}/FasterRCNN_FPN_MobileNetv3_1440_square_{i}', device, res=1440)