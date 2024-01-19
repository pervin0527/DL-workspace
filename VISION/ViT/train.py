import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import logging
import numpy as np

from tqdm import tqdm
from datetime import timedelta
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from apex.parallel import DistributedDataParallel as DDP

from models.configs import vit_configs
from models.model import VisionTransformer

from data.dataset import get_dataloader

from utils.dist_util import get_world_size
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.util import set_seed, read_yaml_file, count_parameters, save_model, AverageMeter


def define_vit(params):
    config = vit_configs(params["model_type"])
    num_classes = 10 if params["dataset"] == "cifar10" else 100
    
    model = VisionTransformer(config, params["img_size"], zero_head=True, num_classes=num_classes)
    model.load_from(np.load(params["pretrained"]))
    model.to(params["device"])
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", params)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    return params, model


def valid(params, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", params["valid_term"])

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=params["local_rank"] not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(params["device"]) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = (all_preds == all_label).mean()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    return accuracy


def train(params, model):
    if params["local_rank"] in [-1, 0]:
        os.makedirs(params["save_dir"], exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(params["save_dir"], "logs", "train"))

    params["train_batch_size"] = params["train_batch_size"] // params["gradient_accumulation_steps"]

    # Prepare dataset
    train_loader, test_loader = get_dataloader(params)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["learning_rate"],
                                momentum=0.9,
                                weight_decay=params["weight_decay"])
    t_total = params["num_steps"]
    if params["decay_type"] == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=params["warmup_steps"], t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=params["warmup_steps"], t_total=t_total)

    # Distributed training
    if params["local_rank"] != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", params["num_steps"])
    logger.info("  Instantaneous batch size per GPU = %d", params["train_batch_size"])
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",params["train_batch_size"] * params["gradient_accumulation_steps"] * (
                    torch.distributed.get_world_size() if params["local_rank"] != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", params["gradient_accumulation_steps"])

    model.zero_grad()
    set_seed(params)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=params["local_rank"] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(params["device"]) for t in batch)
            x, y = batch
            loss = model(x, y)

            if params["gradient_accumulation_steps"] > 1:
                loss = loss / params["gradient_accumulation_steps"]

            loss.backward()

            if (step + 1) % params["gradient_accumulation_steps"] == 0:
                losses.update(loss.item()*params["gradient_accumulation_steps"])
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val))
                if params["local_rank"] in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

                if global_step % params["valid_term"] == 0 and params["local_rank"] in [-1, 0]:
                    accuracy = valid(params, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(params, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if params["local_rank"] in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main(params):
    if params["local_rank"] == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params["num_gpus"] = torch.cuda.device_count()
    else:
        torch.cuda.set_device(params["local_rank"])
        device = torch.device("cuda", params["local_rank"])
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60)) ## 프로세스 그룹을 초기화.
        params["num_gpus"] = 1 ## 분산 학습 환경에서는 각 프로세스가 하나의 GPU를 사용하므로, num_gpu를 1로 설정.
    
    params["device"] = device
    local_rank, num_gpus = params["local_rank"], params["num_gpus"]
    logger.warning(f"Process rank : {local_rank}, Device : {device}, Num_gpus : {num_gpus}, Distributed Training : {bool(local_rank != -1)}")

    set_seed(params)
    params, model = define_vit(params)
    train(params, model)


if __name__ == "__main__":
    params = read_yaml_file("./config.yaml")
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    main(params)