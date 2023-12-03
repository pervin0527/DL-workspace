import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = "/home/pervinco/Models/transformer_translation"

batch_size = 128
max_seq_len = 256
d_model = 512
num_layers = 6
num_heads = 8
ff_hidden = 2048
drop_prob = 0.1

init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epochs = 1000
clip = 1.0
weight_decay = 5e-4
inf = float("inf")