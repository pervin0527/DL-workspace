from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Moduke):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = ScaleDotProductAttention() ## Multi-Head Self-Attention

        """
        head마다 (d_model, d_k)인 3개의 FC layer가 구성되지만, 8번(num_heads) 계산하기엔 부담.
        Vectorization해서 (d_model, d_model)로 정의해 한 번에 수행.
        """
        self.w_q = nn.Linear(d_model, d_model) ## W_Q weight matrix : (d_embed, d_model)
        self.w_k = nn.Linear(d_model, d_model) ## W_K weight matrix : (d_embed, d_model)
        self.w_v = nn.Linear(d_model, d_model) ## W_V weight matrix : (d_embed, d_model)
        self.w_concat = nn.Linear(d_model, d_model) ## for final output : (d_model, d_model)


    def split(self, tensor):
        """
        tensor : [batch_size, seq_len, d_model]
        
        split process
            1. d_k = d_model // num_heads
            2. (batch_size, seq_len, num_heads, d_k)
            3. (batch_size, num_head, seq_len, d_k) --> Transpose는 self-attention의 입력 shape를 맞춰주기 위함.
        """
        batch_size, length, d_model = tensor.size()

        d_k = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_k).transpose(1, 2)
        return tensor
    
    def concat(self, tensor):
        """
        tensor : [batch_size, num_heads, seq_len, d_k]
        1. [batch_size, seq_len, num_heads, d_k]
        2. contiguous 메모리 내 연속되는 텐서로 변환.
        3. view(contiguous텐서의 reshape)

        return to [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = tensor.size()
        d_model = num_heads * d_k

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)


    def forward(self, q, k, v, mask=None):
        """
        q, k, v는 동일한 SRC Embedded matrix.
        """
        ## 1.Convert SRC to Q, K, V
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        ## 2.Split Tensor number of heads --> d_model // num_heads = d_k
        q, k, v = self.split(q), self.split(k), self.split(v)

        ## 3. Scale Dot Product
        out, attention = self.attention(q, k, v, mask=mask) ## Attention Value : (batch_Size, num_heads, seq_len, d_k), Attention Dist : (batch_size, num_heads, seq_len, seq_len)

        ## 4. concatenate
        out = self.concat(out) ## (batch_size, seq_len, d_model)

        ## 5. forward final FC layer
        out = self.w_concat(out) ## (batch_size, seq_len, d_embed)

        return out