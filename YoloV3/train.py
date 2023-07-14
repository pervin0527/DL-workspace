import os
import time
import math
import torch
import random
import numpy as np
import utils.val as validate
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime

from config import train_opt
from utils.metrics import fitness
from utils.general import colorstr
from utils.general import make_dirs
from models.loss import ComputeLoss
from models.yolo import Model, ModelEMA
from data.dataloaders import create_dataloader
from utils.general import LOGGER, TQDM_BAR_FORMAT, strip_optimizer
from models.utils import check_img_size, smart_optimizer, one_cycle, de_parallel, labels_to_class_weights, EarlyStopping

def train():
    if torch.cuda.is_available():
        DEVICE = torch.device(train_opt.device)
        print(DEVICE, torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu")
        print(DEVICE)

    batch_size = train_opt.hyp["batch_size"]
    num_worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(num_worker))

    w = f"{train_opt.save_dir}/weights"
    make_dirs(w)
    last, best = f"{w}/last.pt", f"{w}/best.pt"

    hyp = train_opt.hyp
    train_path, val_path, names = train_opt.data["train"], train_opt.data["val"], train_opt.data["classes"]

    weights = train_opt.weight_dir
    if weights != None and weights != "":
        ckpt = torch.load(weights, map_location='cpu')
        print("Pretrained Weight Loaded.")
    model = Model(train_opt.model, ch=3, nc=train_opt.num_classes, anchors=train_opt.anchors).to(DEVICE)  # create
    print(model)

    # Freeze
    freeze = train_opt.hyp["freeze"]
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(train_opt.hyp["img_size"], gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    train_opt.hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, train_opt.hyp["optimizer"], train_opt.hyp['lr0'], train_opt.hyp['momentum'], train_opt.hyp['weight_decay'])

    # Scheduler
    if train_opt.hyp["cos_lr"]:
        lf = one_cycle(1, train_opt.hyp['lrf'], train_opt.hyp["epochs"])  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / train_opt.hyp["epochs"]) * (1.0 - train_opt.hyp['lrf']) + train_opt.hyp['lrf']  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model)

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size,
                                              gs,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None,
                                              workers=num_worker,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=0)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class

    val_loader = create_dataloader(val_path,
                                    imgsz,
                                    batch_size,
                                    gs,
                                    hyp=hyp,
                                    cache=None,
                                    rect=True,
                                    rank=-1,
                                    workers=num_worker,
                                    pad=0.5,
                                    prefix=colorstr('val: '))[0]
    
    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps) ## 3
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= train_opt.num_classes / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = train_opt.hyp["label_smoothing"]
    model.nc = train_opt.num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, train_opt.num_classes).to(DEVICE) * train_opt.num_classes  # attach class weights
    model.names = names

    # Start training
    best_fitness, start_epoch = 0.0, 0

    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(train_opt.num_classes)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    stopper, stop = EarlyStopping(patience=train_opt.hyp["patience"]), False
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', train_opt.save_dir)}\n"
                f'Starting training for {train_opt.hyp["epochs"]} epochs...')
    
    for epoch in range(start_epoch, train_opt.hyp["epochs"]):
        model.train()

        mloss = torch.zeros(3, device=DEVICE)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(DEVICE, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if train_opt.hyp["multi_scale"]:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(True):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(DEVICE))  # loss scaled by batch_size
           # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%11s' * 2 + '%11.4g' * 5) % (f'{epoch}/{train_opt.hyp["epochs"] - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == train_opt.hyp["epochs"]) or stopper.possible_stop
        if final_epoch:  # Calculate mAP
            results, maps, _ = validate.run(train_opt.data,
                                            batch_size=batch_size * 2,
                                            imgsz=imgsz,
                                            half=True,
                                            model=ema.ema,
                                            single_cls=False,
                                            dataloader=val_loader,
                                            save_dir=train_opt.save_dir,
                                            plots=False,
                                            compute_loss=compute_loss)
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr

            # Save model
            if final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(train_opt),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if train_opt.save_period > 0 and epoch % train_opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt

        # EarlyStopping
        if stop:
            break

        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    torch.cuda.empty_cache()
    return results

    
if __name__ == "__main__":
    train()