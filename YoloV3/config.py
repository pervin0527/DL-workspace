
class TrainConfig:
    epochs = 100
    batch_size = 32
    img_size = 416
    
    learning_rate = 0.001
    lr_patience = 10
    lr_factor = 0.8

    iou_thres = 0.5
    nms_thres = 0.5
    conf_thres = 0.5

    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    save_dir = "/home/pervinco/Models/YoloV3"
    pretrained_weight_path = ""
    darknet_weight_path = "/home/pervinco/Models/YoloV3/darknet53.conv.74"
    ## "/home/pervinco/Models/YoloV3/yolov3.weights"

class TestConfig:
    batch_size = 16
    img_size = 416
    iou_thres = 0.5
    nms_thres = 0.5
    conf_thres = 0.5
    weight_path = ""
    darknet_weight_path = "/home/pervinco/Models/YoloV3/yolov3.weights"

train_cfg = TrainConfig()
test_cfg = TestConfig()