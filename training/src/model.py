import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model =torchvision.models.detection.fasterrcnn_resnet50_fpn(weight="DEFAULT")   #incarcarea modelului
    in_features =model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor =FastRCNNPredictor(in_features,num_classes+1)   #clasa 0 , background

    return model