from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, init_lr, max_lr, min_lr, total_epochs, warmup_epochs, last_epoch=-1):
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        super(LinearWarmupDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.total_epochs <= self.warmup_epochs:
            raise ValueError("Total epochs must be greater than warmup epochs")

        if self.last_epoch < self.warmup_epochs:
            # 워밍업 단계 - init_lr에서 max_lr까지 선형 증가
            lr = self.init_lr + (self.max_lr - self.init_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            # 선형 감소 단계 - max_lr에서 min_lr까지 선형 감소
            decay_epochs = self.total_epochs - self.warmup_epochs
            if decay_epochs > 0:
                lr = self.max_lr - (self.max_lr - self.min_lr) * ((self.last_epoch - self.warmup_epochs) / decay_epochs)
            else:
                lr = self.min_lr
        return [lr for base_lr in self.base_lrs]
