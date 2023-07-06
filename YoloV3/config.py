
class TrainConfig:
    epochs = 1000
    batch_size = 64
    train_img_size = 256
    
    learning_rate = 0.001
    step_size = 10
    gamma = 0.8

    iou_thres = 0.5
    nms_thres = 0.5
    conf_thres = 0.5

    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    save_dir = "/home/pervinco/Models/YoloV3"

train_cfg = TrainConfig()