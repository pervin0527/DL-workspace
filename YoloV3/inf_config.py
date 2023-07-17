class TestConfig:
    weights = "/home/pervinco/Models/YoloV3/runs/train/exp/weights/best.pt"
    source = "./dog.jpg"
    data_root = "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit/VOC2012/yolo"
    data = {"path" : data_root,
            "train" : f"{data_root}/train",
            "val" : f"{data_root}/val",
            "names" : [x.strip() for x in open("./classes.txt", "r").readlines()]}
    data["names"] = {idx : name for idx, name in enumerate(data["names"])}
    data.update({"nc" : len(data["names"])})

    device = 0
    imgsz = 416
    iou_thres = 0.45
    conf_thres = 0.25
    max_det = 1000