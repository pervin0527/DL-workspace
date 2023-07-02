import os
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import train_cfg
from data.coco_dataset import CoCoDataset
from model.evaluate_utils import evaluate
from model.train_utils import get_model, train_one_epoch, write_tb
from data.coco_utils import plot_loss_and_lr, plot_map
from data.pascal_voc_dataset import PascalVocDataset
from data.data_utils import Compose, ToTensor, RandomHorizontalFlip

if __name__ == "__main__":
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    if not os.path.exists(train_cfg.save_dir):
        os.makedirs(train_cfg.save_dir)
    writer = SummaryWriter(f"{train_cfg.save_dir}/logs")

    data_transform = {
        "train" : Compose([ToTensor(), 
                           RandomHorizontalFlip(p=0.5)]),
        "valid" : Compose([ToTensor()])
    }

    train_dataset = PascalVocDataset(train_cfg.data_dir, "trainval", data_transform["train"])
    valid_dataset = PascalVocDataset(train_cfg.data_dir, "val", data_transform["valid"])
    num_worker = min([os.cpu_count(), train_cfg.batch_size if train_cfg.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(num_worker))

    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=num_worker, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_cfg.batch_size, num_workers=num_worker, collate_fn=valid_dataset.collate_fn)

    model = get_model(num_classes=len(train_dataset.classes))
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=train_cfg.lr,
                                momentum=train_cfg.momentum, weight_decay=train_cfg.weight_decay)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=train_cfg.lr_dec_step_size,
                                                   gamma=train_cfg.lr_gamma)

    # train from pretrained weights
    if train_cfg.resume != "":
        checkpoint = torch.load(train_cfg.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        train_cfg.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(train_cfg.start_epoch))

    train_loss = []
    learning_rate = []
    train_mAP_list = []
    val_mAP = []

    best_mAP = 0
    for epoch in range(train_cfg.start_epoch, train_cfg.num_epochs):
        loss_dict, total_loss = train_one_epoch(model, optimizer, train_dataloader,
                                                device, epoch, train_loss=train_loss, train_lr=learning_rate,
                                                print_freq=50, warmup=False)

        lr_scheduler.step()

        print("------>Starting training data valid")
        _, train_mAP = evaluate(model, train_dataloader, device=device, mAP_list=train_mAP_list)

        print("------>Starting validation data valid")
        _, mAP = evaluate(model, valid_dataloader, device=device, mAP_list=val_mAP)
        print('training mAp is {}'.format(train_mAP))
        print('validation mAp is {}'.format(mAP))
        print('best mAp is {}'.format(best_mAP))

        board_info = {'lr': optimizer.param_groups[0]['lr'],
                      'train_mAP': train_mAP,
                      'val_mAP': mAP}

        for k, v in loss_dict.items():
            board_info[k] = v.item()
        board_info['total loss'] = total_loss.item()
        write_tb(writer, epoch, board_info)

        if mAP > best_mAP:
            best_mAP = mAP
            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            model_save_dir = train_cfg.model_save_dir
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(save_files,
                       os.path.join(model_save_dir, "{}-model-{}-mAp-{}.pth".format(train_cfg.backbone, epoch, mAP)))
    writer.close()
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, train_cfg.model_save_dir)

    # plot mAP curve
    if len(val_mAP) != 0:
        plot_map(val_mAP, train_cfg.model_save_dir)