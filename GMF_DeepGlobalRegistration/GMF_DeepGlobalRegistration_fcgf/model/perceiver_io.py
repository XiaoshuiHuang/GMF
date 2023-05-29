from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    '''
        x : 表示中间隐式数组
        kwargs : 多余参数
            'context' - 表示原数组
    '''
    def forward(self, x, **kwargs):
        x = self.norm(x)
        # 判断是否存在'context'，若不存在，则说明是self-attention，若存在，说明是cross-attention
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)
# 注意力网络
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        # QK除的比例，目的是计算softmax数值不会太大
        self.scale = dim_head ** -0.5
        # attention的头大小
        self.heads = heads
        # 产生Q以及K，V的线性层
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    # 进行，attention
    def forward(self, x, context = None, mask = None):
        h = self.heads      # attention的头维度大小
        # 通过query数组求MLP得到Q数组
        q = self.to_q(x)
        context = default(context, x)       # 原数组，若context不存在，则使用Query数组（说明是self-attention）
        temp = self.to_kv(context)          # 通过对原组进行MLP，求出query维度 * 2的向量temp
        k,v = temp.chunk(2, dim = -1)       # 得到向量temp进行分裂，生成key和value
        # 根据cross_attention的头数量，来重塑Q,K,V的形状
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # 爱因斯坦求和，本质是矩阵相乘，目的是求出Q*K，并乘以1/Query的维度开根号，以缩小取值，得到Q和K的比例分数
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # attention, what we cannot get enough of，得到QK的softmax分数
        attn = sim.softmax(dim = -1)
        # 再将QK(softmax)与V矩阵相乘，求出QKV
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# Convolutional Position Encoding
class ConvPosEnc(nn.Module):
    def __init__(self, dim_q, dim_content, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj_q = nn.Conv1d(
            in_channels=dim_q,
            out_channels=dim_q,
            kernel_size=k,
            stride=1,
            padding=k//2,
            groups=dim_q
        )

        self.proj_content = nn.Conv1d(
            in_channels=dim_content,
            out_channels=dim_content,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim_content
        )

    def forward(self,q,content):
        q = q.permute(0,2,1)
        q = self.proj_q(q) + q
        q = q.permute(0,2,1)

        # B,C,H,W = content.shape
        content = content.permute(0, 2, 1)
        content = self.proj_content(content) + content
        content = content.permute(0,2,1)

        return q,content

# main class
class PerceiverIO(nn.Module):
    def __init__(
        self,
        depth,                                  # Process的层数，可以设置为0
        dim,                                    # image dim
        latent_dim = 512,                       # PC dim
        cross_heads = 1,                        # Encoder的头大小
        latent_heads = 8,                       # Process的头大小
        cross_dim_head = 64,                    # Encoder和Decoder中计算过程中的维度
        latent_dim_head = 64,                   # Process计算过程中的维度
        weight_tie_layers = False,
        pe=False
    ):
        super().__init__()

        self.pe = pe
        if(pe):
            # position encoding
            self.cpe = ConvPosEnc(
                dim_q=latent_dim,
                dim_content=dim
            )

        # Encoder部分，不一定是一层
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        # self.cross_attend_blocks =  PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim)
        # Process部分，多层
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))
        # # Decoder部分，只能为一层
        # self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        # self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        # # 将最后结果进行重塑为指定想要的形状
        # self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,                           # Image特征
        mask = None,
        queries_encoder = None,         # PointCloud特征
    ):
        b, *_, device = *data.shape, data.device
        # PointCloud特征
        x = queries_encoder

        # cpe
        if(self.pe):
            x,data = self.cpe(
                q=x,
                content=data,
            )

        # ---- Encoder过程 ----
        cross_attn, cross_ff = self.cross_attend_blocks
        # cross_attn = self.cross_attend_blocks
        # 经过Attention得到Query与原Query相加，维度不变
        x = cross_attn(x, context = data, mask = mask) + x
        # x = cross_attn(x, context = data, mask = mask)
        # 其目的是让特征较重要的部分，更加重要
        x = cross_ff(x) + x
        # ---- Encoder过程 ----


        #  ---- Process过程 ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        # ---- Process过程 ----

        return x

        # # ---- Decoder过程 ----
        # if not exists(queries_decoder):
        #     return x
        # # cross attend from decoder queries to latents
        # latents = self.decoder_cross_attn(queries_decoder, context = x)
        # # optional decoder feedforward
        # if exists(self.decoder_ff):
        #     latents = latents + self.decoder_ff(latents)
        # # final linear out
        # return self.to_logits(latents)
        # # ---- Decoder过程 ----
