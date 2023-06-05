import torch
import torchvision.transforms as transforms

from tqdm import tqdm
from torch import nn
from model import PlainNet
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from preprocessing import get_mean_std, ScaleJitter

def valid(dataloader, model, loss_fn):
    model.eval()
    valid_loss, valid_correct = 0, 0
    with torch.no_grad():
        for iter_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()

            valid_loss += loss * X.size(0)
            _, pred = torch.max(pred, 1)
            _, y = torch.max(y, 1)        
            valid_correct += (pred == y).sum().item()

    valid_loss /= len(dataloader.dataset)
    valid_accuracy = valid_correct / len(dataloader.dataset)
    print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f} \n")

    return valid_loss, valid_accuracy


def train(dataloader, model, loss_fn, optimizer):
    global lr_patience, early_stop_patience
    best_loss = 0
    writer = SummaryWriter(log_dir=LOG_PATH)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0, 0
        pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1} / {EPOCHS}', unit='step')

        for iter_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            ## prediction & loss
            pred = model(X)
            loss = loss_fn(pred, y)

            ## Backprop
            optimizer.zero_grad() ## 각 epoch마다 gradient를 측정하기 위해 0으로 초기화한다.
            loss.backward() ## 현재 loss값에 대한 backpropagation을 시작한다.
            optimizer.step() ## parameter를 update한다.

            _, pred = torch.max(pred, 1)
            _, y = torch.max(y, 1)
            correct = (pred == y).sum().item()

            train_loss += loss.item() * X.size(0)
            train_correct += correct

            pbar.set_postfix({"Loss" : loss.item(), "Acc" : correct / X.size(0)})
            pbar.update(1)

        pbar.close()
        train_loss /= len(dataloader.dataset)
        train_accuracy = train_correct / len(dataloader.dataset)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        ## Validation step
        valid_loss, valid_accuracy = valid(valid_dataloader, model, loss_fn)

        ## Record train & valid & lr log to tensorboard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", valid_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/Validation", valid_accuracy, epoch)

        writer.add_scalars("Loss", {"train_loss" : train_loss, "valid_loss" : valid_loss}, epoch)
        writer.add_scalars("Accuracy", {"train_accuracy" : train_accuracy, "valid_accuracy" : valid_accuracy}, epoch)

        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)


        ## LR scheduler & Early Stopping
        if epoch == 0:
            best_loss = valid_loss
        else:
            if valid_loss <= best_loss:
                print(f"Valid loss decreased. The minimum valid loss updated {best_loss:.4f} to {valid_loss:.4f}.")
                best_loss = valid_loss
                lr_patience = 10
            else:
                lr_patience -= 1
                print(f"Valid loss did not decrease. patience : {lr_patience} | best : {best_loss:.4f} | current : {valid_loss:.4f}")

                if lr_patience == 0:
                    lr_patience = 10
                    early_stop_patience -= 1
                    new_lr = optimizer.param_groups[0]["lr"] * 0.1

                    print(f"Early Stop patience : {early_stop_patience}, learning rate changed {new_lr * 10} to {new_lr}")
                    optimizer.param_groups[0]['lr'] = new_lr

                if early_stop_patience == 0:
                    print("Early stopping patience is 0. Train stopped.")
                    break

    writer.close()
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"train finished. pth file saved at {SAVE_PATH}")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    ## Hyper-parameters
    TRAIN_MODEL_NAME = "PlainNet34"
    EPOCHS = 1000
    IMG_SIZE = 224
    BATCH_SIZE = 256
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9

    ## Dir
    ROOT = "/home/pervinco"
    DATASET_NAME = "sports"
    DATASET_PATH = f"{ROOT}/Datasets/{DATASET_NAME}"

    PTH_NAME = f"{TRAIN_MODEL_NAME}/{DATASET_NAME}"
    SAVE_PATH = f"{ROOT}/Models/ResNet/{PTH_NAME}.pth"
    LOG_PATH = f"{ROOT}/Models/ResNet/{PTH_NAME}"

    ## Data Processing
    mean, std = get_mean_std(f"{DATASET_PATH}/train")
    train_transform = transforms.Compose([
        ScaleJitter(min_size=256, max_size=480, crop_size=(224, 224), p=0.5),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE), antialias=False),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.5, 0.5)),
        transforms.Normalize(mean=mean, std=std),
    ])

    ## Dataloader
    train_dataset = CustomDataset(DATASET_PATH, "train", train_transform)
    valid_dataset = CustomDataset(DATASET_PATH, "valid", train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32)

    print(f"Total train data : {len(train_dataloader.dataset)}, step numbers : {len(train_dataloader)}")
    print(f"Total test data : {len(valid_dataloader.dataset)}, step numbers : {len(valid_dataloader)} \n")

    ## Build Model
    classes = train_dataset.get_classes()
    model = PlainNet(len(classes), init_weights=True)
    print(f"{TRAIN_MODEL_NAME} \n {model}")

    ## Loss func & Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE, 
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    
    train(train_dataloader, model, loss_func, optimizer)