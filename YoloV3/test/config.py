class TrainConfig:
    root = "/home/pervinco"
    weight_dir = None ## f"{root}/Models/YoloV3/yolov3.pt"
    save_dir = f"{root}/Models/YoloV3"
    device = "cuda:0"

    data = {
        "train" : f"{root}/Datasets/PASCAL_VOC/VOCdevkit/VOC2012/yolo/train",
        "val" : f"{root}/Datasets/PASCAL_VOC/VOCdevkit/VOC2012/yolo/val",
        "classes" : [x.strip() for x in open("./classes.txt")]
    }
    num_classes = len(data["classes"])
    save_period = -1

    hyp = {
        "batch_size" : 32,
        "epochs" : 100,
        "img_size" : 640,
        "optimizer" : "SGD",
        "patience" : 100,
        "multi_scale" : False,
        "cos_lr" : False,
        "lr0": 0.01,
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
        "copy_paste": 0.0,
        "freeze" : [0],
        "label_smoothing" : 0.0
    }

    anchors = [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
    model = {
       "depth_multiple" : 1.0,
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
                  [-1, 1, "Conv", [1024, 3, 1]],
                  
                  [-2, 1, "Conv", [256, 1, 1]],
                  [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
                  [[-1, 8], 1, "Concat", [1]],
                  [-1, 1, "Bottleneck", [512, False]],
                  [-1, 1, "Bottleneck", [512, False]],
                  [-1, 1, "Conv", [256, 1, 1]],
                  [-1, 1, "Conv", [512, 3, 1]],
                  
                  [-2, 1, "Conv", [128, 1, 1]],
                  [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
                  [[-1, 6], 1, "Concat", [1]],
                  [-1, 1, "Bottleneck", [256, False]],
                  [-1, 2, "Bottleneck", [256, False]],
                  
                  [[27, 22, 15], 1, "Detect", [num_classes, anchors]], 
        ]
    }
    cache = None

train_opt = TrainConfig()