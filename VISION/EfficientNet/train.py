import os
import yaml
import time
import torch

from torch import nn
from torch.utils.data import DataLoader

from data.dataloader import MyDataset
from models.model import EfficientNet
from utils.utils import adjust_learning_rate, accuracy, AverageMeter, ProgressMeter

def train(dataloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(dataloader), batch_time, data_time, losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            progress.print(i)

if __name__ == "__main__":

    with open("config.yaml", "r") as f:
           config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    model_name= config["model_name"]
    if config["pretrained"]:
        model = EfficientNet.from_pretrained(model_name, advprop=config["advprop"])
        print(f"=> using pre-trained model {model_name}")
    else:
        model = EfficientNet.from_name(model_name)
        print(f"=> creating model {model_name}")

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), config["learning_rate"],
                                momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    
    img_size = EfficientNet.get_image_size(model_name)
    train_dataset = MyDataset(data_dir=config["data_dir"], set_name=config["train_set"], img_size=img_size)
    valid_dataset = MyDataset(data_dir=config["data_dir"], set_name=config["valid_set"], img_size=img_size)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"])
    
    start_epoch = config["start_epoch"]
    retraining = config["resume"]
    if retraining:
         if os.path.isfile(retraining):
              print(f"=> Loading checkpoint {retraining}")
              checkpoint = torch.load(retraining)

              start_epoch = checkpoint["epoch"]
              best_acc1 = checkpoint["best_acc1"]
              best_acc1 = best_acc1.to(device)

              model.load_state_dict(checkpoint["state_dict"])
              optimizer.load_state_dict(checkpoint["optimizer"])

              print(f"=> loaded checkpoint {retraining} epoch : {start_epoch}")
    else:
         print(f"=> no checkpoint found at {retraining}")

    epochs = config["epochs"]
    for epoch in range(start_epoch, epochs):
         adjust_learning_rate(optimizer, epoch, config["learning_rate"])

         train(train_dataloader, model, criterion, optimizer, epoch)