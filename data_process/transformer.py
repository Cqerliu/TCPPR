import torch.nn.functional as F
from torch import nn, einsum
from entmax import Sparsemax
from einops import rearrange

# helpers 来判断传入的值是否为NONE
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


# classes
# Residual block 残差连接模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# PreNorm block 层归一化模块：先层归一化然后残差连接
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

# GEGLU as the activation function
# 将输入张量x在最后一个维度上分成两部分，分别赋值给x和gates。然后将x与经过gelu激活函数处理后的gates逐元素相乘并返回
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# 前馈网络层 创建一个包含线性层、GEGLU激活函数、Dropout层和另一个线性层的序列模型
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads = 8,
            dim_head = 16,
            dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        # self.selector = nn.Softmax(dim = -1)
        # self.selector = Entmax15(dim = -1)
        self.selector = Sparsemax(dim = -1)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.selector(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)


# transformer

class PrompterTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim, # 嵌入维度
                 output_dim,
                 depth,
                 heads,
                 dim_head,
                 attn_dropout, # 注意力层的 Dropout 概率
                 ff_dropout): # 前馈层的 Dropout 概率
        super().__init__()
        # 创建一个嵌入层nn.Embedding用于将输入的整数索引转换为嵌入向量，创建一个空的模块列表self.layers用于存储变压器层。
        self.embeds = nn.Embedding(input_dim, embedding_dim) # 将输入的整数索引转换为嵌入向量
        self.layers = nn.ModuleList([]) # 存储 Transformer 层

        # transformer layers
        # 通过循环创建指定深度的变压器层，每个层包含一个注意力模块和一个前馈模块，都使用了残差连接和层归一化。
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(
                    PreNorm(embedding_dim, Attention(embedding_dim, heads = heads, dim_head = dim_head,
                                                     dropout = attn_dropout))),
                Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, dropout = ff_dropout))),
            ]))

        # 创建一个输出层，包含展平操作、Dropout层和线性层，用于将模型的输出映射到指定的输出维度
        self.output_layer = nn.Sequential(nn.Flatten(-2, -1), # 将张量展平
                                          nn.Dropout(p = 0.5),
                                          nn.Linear(79 * embedding_dim, output_dim)) # 线性层，将展平后的张量映射到输出维度

# 首先将输入x通过嵌入层得到嵌入向量，然后依次通过每个变压器层的注意力模块和前馈模块，最后通过输出层得到最终的输出结果
    def forward(self, x):
        x = self.embeds(x) # (B, token_dims) -> (B, token_dims, embedding_dim)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x



