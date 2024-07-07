from torch import nn
from torchvision.models import resnet50, mobilenet_v2, MobileNet_V2_Weights, ResNet50_Weights
from torchvision.ops.misc import FrozenBatchNorm2d

from .utils import pooling


class RCNN_ResNet50(nn.Module):

    def __init__(self, roi_res=100, pooling_type='square'):
        super().__init__()
        # load backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=2)
        
        # freeze bottom layers
        layers_to_train = ['layer4', 'layer3', 'layer2']
        for name, parameter in self.backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        
        # ROI pooling
        self.roi_res = roi_res
        self.pooling_type = pooling_type
        
    def forward(self, image, rois):
        # pool ROIs from image
        warps = pooling.roi_pool(image, rois, self.roi_res, self.pooling_type)
        
        # pass warped images through classifier
        class_logits = self.backbone(warps)
        
        return class_logits


class RCNN_MobileNetv2(nn.Module):

    def __init__(self, roi_res=100, pooling_type='square'):
        super().__init__()
        # load backbone
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        # Replace the last fully connected layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 2)
        
        
        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze specific layers
        layers_to_train = ['4', '3', '2']
        for name, module in self.backbone.features.named_children():
            if name in layers_to_train:
                for param in module.parameters():
                    param.requires_grad = True
        

        # ROI pooling
        self.roi_res = roi_res
        self.pooling_type = pooling_type
        
    def forward(self, image, rois):
        # pool ROIs from image
        warps = pooling.roi_pool(image, rois, self.roi_res, self.pooling_type)
        
        # pass warped images through classifier
        class_logits = self.backbone(warps)
        
        return class_logits

