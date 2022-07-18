import haiku as hk

import jax
import jax.numpy as jnp
from jax import einsum

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)

def default(val, d):
    return val if exists(val) else d

# positional embedding

class RotaryEmbedding(hk.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (jax.arange(0, dim, 2).float() / dim))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, offset = 0):
        seq = jnp.arange(max_seq_len) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis = -1)
        return rearrange(emb, 'n d -> 1 1 n d')

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return jnp.concatenate((-x2, x1), axis = -1)

def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * jnp.cos(freqs)) + (rotate_half(t) * jnp.sin(freqs))
    return jnp.concatenate((t, t_pass), axis = -1)

# attention

class Attention(hk.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.dropout = dropout

        self.to_q = hk.Linear(inner_dim, bias = False)
        self.to_k = hk.Linear(inner_dim, bias = False)
        self.to_v = hk.Linear(inner_dim, bias = False)
        self.to_out = hk.Linear(dim)

    def forward(self, x, context = None, pos_emb = None):
        b, h, scale = x.shape[0], self.heads, self.scale

        kv_input = default(context, x)

        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * scale

        # apply relative positional encoding (rotary embeddings)

        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)

            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # derive query key similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking

        mask_value = -jnp.finfo(sim.dtype).max

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = jnp.ones(i, j, dtype = np.bool).triu(j - i + 1)
            sim = jnp.where(causal_mask, sim, mask_value)

        # attention

        attn = jax.nn.softmax(sim, axis = -1)

        attn = hk.dropout(rng = hk.next_rng_key(), rate = self.dropout, x = attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine heads linear out

        return self.to_out(out)

