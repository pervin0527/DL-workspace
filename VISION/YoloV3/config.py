class TrainConfig:
    model_root = "/home/pervinco/Models/YoloV3"
    model = {"weights" : f"{model_root}/yolov3.pt",
             "folder" : "runs/train",
             "name" : "exp"}

    data_root = "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit/VOC2012/yolo"
    data = {"path" : data_root,
            "train" : f"{data_root}/train",
            "val" : f"{data_root}/val",
            "names" : [x.strip() for x in open("./classes.txt", "r").readlines()]}
    data["names"] = {idx : name for idx, name in enumerate(data["names"])}
    data.update({"nc" : len(data["names"])})

    resume = False
    latest_weight = "/home/pervinco/Models/YoloV3/yolov3.pt"

    device = 0
    epochs = 100
    batch_size = 1
    imgsz = 416
    optimizer = "SGD"
    patience = 100
    save_period = -1
    freeze = [0]
    cos_lr = False
    sync_bn = False
    hyp = {"lr0": 0.01,
           "lrf": 0.01,
           "momentum": 0.937,
           "weight_decay": 0.0005,
           "warmup_epochs": 3.0,
           "warmup_momentum": 0.8,
           "warmup_bias_lr": 0.1,
           "box": 0.05,
           "cls": 0.5,
           "cls_pw": 1.0,
           "obj": 1.0,
           "obj_pw": 1.0,
           "iou_t": 0.20,
           "anchor_t": 4.0,
           "fl_gamma": 0.0,
           "hsv_h": 0.015,
           "hsv_s": 0.7,
           "hsv_v": 0.4,
           "degrees": 0.0,
           "translate": 0.1,
           "scale": 0.5,
           "shear": 0.0,
           "perspective": 0.0,
           "flipud": 0.0,
           "fliplr": 0.5,
           "mosaic": 1.0,
           "mixup": 0.0,
           "copy_paste": 0.0}
    
    anchors = [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
    cfg = {"depth_multiple" : 1.0,
           "width_multiple" : 1.0,
           "backbone" : [[-1, 1, "Conv", [32, 3, 1]],
                         [-1, 1, "Conv", [64, 3, 2]],
                         [-1, 1, "Bottleneck", [64]],
                         [-1, 1, "Conv", [128, 3, 2]],
                         [-1, 2, "Bottleneck", [128]],
                         [-1, 1, "Conv", [256, 3, 2]],
                         [-1, 8, "Bottleneck", [256]],
                         [-1, 1, "Conv", [512, 3, 2]],
                         [-1, 8, "Bottleneck", [512]],
                         [-1, 1, "Conv", [1024, 3, 2]],
                         [-1, 4, "Bottleneck", [1024]]],

            "head" : [[-1, 1, "Bottleneck", [1024, False]],
                      [-1, 1, "Conv", [512, 1, 1]],
                      [-1, 1, "Conv", [1024, 3, 1]],
                      [-1, 1, "Conv", [512, 1, 1]],
                      [-1, 1, "Conv", [1024, 3, 1]],  # 15 (P5/32-large)
                      [-2, 1, "Conv", [256, 1, 1]],
                      [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
                      [[-1, 8], 1, "Concat", [1]],  # cat backbone P4
                      [-1, 1, "Bottleneck", [512, False]],
                      [-1, 1, "Bottleneck", [512, False]],
                      [-1, 1, "Conv", [256, 1, 1]],
                      [-1, 1, "Conv", [512, 3, 1]],  # 22 (P4/16-medium)
                      [-2, 1, "Conv", [128, 1, 1]],
                      [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
                      [[-1, 6], 1, "Concat", [1]],  # cat backbone P3
                      [-1, 1, "Bottleneck", [256, False]],
                      [-1, 2, "Bottleneck", [256, False]],  # 27 (P3/8-small)
                      [[27, 22, 15], 1, "Detect", [data["nc"], anchors]]]}
    cfg.update({"anchors" : anchors})

    seed = 0
    noval = False
    nosave = False
    noplots = False
    evolve = None
    single_cls = False
    cache = False
    rect = False
    image_weights = False
    quad = False
    noautoanchor = False
    label_smoothing = False
    multi_scale = False
    
if __name__ == "__main__":
    opt = TrainConfig()
    print(opt)