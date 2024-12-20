times['R-CNN (256)'] = time_model(RCNN_MobileNetv2(roi_res=256), res=[4000, 3000])
# print("times 256 defined")

# times['Faster R-CNN FPN (800)'] = time_model(FasterRCNN_FPN_MobileNetv2(), res=[1067, 800])
# print("times 800 defined")

# times['Faster R-CNN FPN (1100)'] = time_model(FasterRCNN_FPN_MobileNetv2(), res=[1467, 1100])
# print("times 1100 defined")

# times['Faster R-CNN FPN (1440)'] = time_model(FasterRCNN_FPN_MobileNetv2(), res=[1920, 1440])
# print("times 1440 defined")