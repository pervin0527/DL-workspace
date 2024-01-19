# Vision Transformer

[ViT](https://arxiv.org/abs/2010.11929)논문을 보고 만든 저장소입니다.

제가 읽으며 정리한 내용은 제 [블로그](https://pervin0527.notion.site/ViT-AN-IMAGE-IS-WORTH-16X16-WORDS-TRANSFORMERS-FOR-IMAGE-RECOGNITION-AT-SCALE-d7890b5b08774289bb73740c1041f59c?pvs=4)에서 볼 수 있습니다.

## ImageNet21k Pretrained Weights

     wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz
     wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz
     wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz
     wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz
     wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

## ImageNet21k + ImageNet2012 Pretrained Weights

    wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_32.npz
    wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_16.npz
    wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_16-224.npz


## Results
### imagenet-21k

|    model     |  dataset  | resolution | acc(official) | acc(this repo) |  time   |
|:------------:|:---------:|:----------:|:-------------:|:--------------:|:-------:|
|   ViT-B_16   | CIFAR-10  |  224x224   |       -       |     0.9908     | 3h 13m  |
|   ViT-B_16   | CIFAR-10  |  384x384   |    0.9903     |     0.9906     | 12h 25m |
|   ViT_B_16   | CIFAR-100 |  224x224   |       -       |     0.923      |  3h 9m  |
|   ViT_B_16   | CIFAR-100 |  384x384   |    0.9264     |     0.9228     | 12h 31m |
| R50-ViT-B_16 | CIFAR-10  |  224x224   |       -       |     0.9892     | 4h 23m  |
| R50-ViT-B_16 | CIFAR-10  |  384x384   |     0.99      |     0.9904     | 15h 40m |
| R50-ViT-B_16 | CIFAR-100 |  224x224   |       -       |     0.9231     | 4h 18m  |
| R50-ViT-B_16 | CIFAR-100 |  384x384   |    0.9231     |     0.9197     | 15h 53m |
|   ViT_L_32   | CIFAR-10  |  224x224   |       -       |     0.9903     | 2h 11m  |
|   ViT_L_32   | CIFAR-100 |  224x224   |       -       |     0.9276     |  2h 9m  |
|   ViT_H_14   | CIFAR-100 |  224x224   |       -       |      WIP       |         |


### imagenet-21k + imagenet2012

|    model     |  dataset  | resolution |  acc   |
|:------------:|:---------:|:----------:|:------:|
| ViT-B_16-224 | CIFAR-10  |  224x224   |  0.99  |
| ViT_B_16-224 | CIFAR-100 |  224x224   | 0.9245 |
|   ViT-L_32   | CIFAR-10  |  224x224   | 0.9903 |
|   ViT-L_32   | CIFAR-100 |  224x224   | 0.9285 |


### shorter train
* In the experiment below, we used a resolution size (224x224).

|  upstream   |  model   |  dataset  | total_steps /warmup_steps | acc(official) | acc(this repo) |
|:-----------:|:--------:|:---------:|:-------------------------:|:-------------:|:--------------:|
| imagenet21k | ViT-B_16 | CIFAR-10  |          500/100          |    0.9859     |     0.9859     |
| imagenet21k | ViT-B_16 | CIFAR-10  |         1000/100          |    0.9886     |     0.9878     |
| imagenet21k | ViT-B_16 | CIFAR-100 |          500/100          |    0.8917     |     0.9072     |
| imagenet21k | ViT-B_16 | CIFAR-100 |         1000/100          |    0.9115     |     0.9216     |
