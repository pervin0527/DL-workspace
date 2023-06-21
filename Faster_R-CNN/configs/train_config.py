class Confing:
    ## data dir
    DATA_PATH = "/home/pervinco/Datasets/COCO"

    ## backbone model
    BACKBONE = "resnet50"

    ## hyper paramters
    BATCH_SIZE = 6

    ## image scale, normalize
    MIN_SIZE = 800
    MAX_SIZE = 1000
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # anchor parameters
    ANCHOR_SIZE = [64, 128, 256]
    ANCHOR_RATIO = [0.5, 1, 2.0]

train_cfg = Confing()