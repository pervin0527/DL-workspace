import math
from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, num_heads, seq_len, d_k = k.size()

        ## 1. Dot Product Q with K_transpose --> Attention Score
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_k) ## (batch_Size, num_heads, seq_len, seq_len)

        ## 2. Pad Masking
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000) ## mask_shape : (batch_size, seq_len, seq_len)

        ## 3. Attention Distribution
        score = self.softmax(score) ## (batch_size, num_heads, seq_len, seq_len)

        ## 4. Attention Value multiply with V
        v = score @ v ## (batch_Size, num_heads, seq_len, d_k)

        return v, score
