class TrainConfig:
    save_dir = "/home/pervinco/Models/Faster-RCNN"
    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    
    device = "cuda:0"
    backbone_name = "vgg16"
    resume = ""
    
    start_epoch = 0
    num_epochs = 5000
    batch_size = 4
    lr = 5e-3
    momentum = 0.9
    weight_decay = 0.0005
    lr_gamma = 0.33
    lr_dec_step_size = 100

    min_size = 600
    max_size = 1000
    anchor_scale = [64, 128, 256]
    aspect_ratio = [0.5, 1, 2.0]
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5

    rpn_nms_thresh = 0.7
    rpn_pre_nms_top_n_train = 2000
    rpn_post_nms_top_n_train = 2000
    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_test = 1000

    roi_out_size = [7, 7]
    roi_sample_rate = 2

    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None

train_cfg = TrainConfig()