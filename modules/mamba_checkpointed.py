import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
import math
from modules.mamba_utils import RMSNorm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from transformers import BertForMaskedLM
from causal_conv1d import causal_conv1d_fn


device = torch.device("cuda")

# @triton.jit
# def generate_a(A, d_in, d_hidden, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     tl.device_print("pid", pid)
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < d_in * d_hidden

#     # loopy doopy because there is no parallel filter function / indexing in triton bruh :(
#     for offset in offsets: 
#         x = offset // d_in
#         y = offset - (x * d_in)
#         value = tl.sqrt(2 * x + 1) * tl.sqrt(2 * y + 1) if x > y else tl.zeros(1, dtype=tl.float32) if x < y else x + 1
#         tl.store(A + offset, value, mask=mask)

def generate_a(d_in: int, d_hidden: int):
    t = torch.empty(d_in, d_hidden).to(device=device)
    for i in range(d_in):
        for j in range(d_hidden):
            if i == j: t[i][j] = i + 1
            elif i > j: t[i][j] = math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
            else: t[i][j] = 0
    
    return t


class S4Checkpointed(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dt_rank: int, d_memory: int, checkpoint_step: int = 16):
        super().__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dt_rank = dt_rank
        self.d_memory = d_memory
        self.step = checkpoint_step

        # BLOCK_SIZE = 32

        A = repeat(
            torch.arange(1, self.d_hidden + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_in,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_in, device=device, dtype=torch.float))  # Keep in fp32
        self.D._no_weight_decay = True

        self.x_proj = nn.Linear(d_in, 3 * d_hidden + 2 * d_in + 2 * dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_in, bias=True)
        self.dt_local_proj = nn.Linear(dt_rank, d_in, bias=True)

        dt_max = 0.1
        dt_min = 1e-3
        dt_init_floor = 1e-4

        dt = torch.exp(
            torch.rand(d_in) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_local_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        self.dt_local_proj.bias._no_reinit = True

        self.w_K = nn.Linear(d_in, d_memory)
        self.w_Q = nn.Linear(d_in, d_memory)
        self.w_V = nn.Linear(d_in, d_memory)
        self.w_O = nn.Linear(d_memory, d_in)


    # refactor all these rearranges later haha
    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        A = -torch.exp(self.A_log.float())
        bcd = rearrange(self.x_proj(x), "b l d -> b d l")
        (B, B_local, C, E, F, dt, dt_local) = bcd.split(split_size=[self.d_hidden, self.d_hidden, self.d_hidden, self.d_in, self.d_in, self.dt_rank, self.dt_rank], dim=-2)
        delta = rearrange(self.dt_proj(rearrange(dt, "b d l -> b l d")), "b l d -> b d l")
        delta_local = rearrange(self.dt_local_proj(rearrange(dt_local, "b d l -> b l d")), "b l d -> b d l")

        y = selective_scan_fn(rearrange(x, "b l d -> b d l"), delta, A, B, C, self.D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
        # y_local = selective_scan_fn(rearrange(x, "b l d -> b d l"), delta_local, A, B_local, C=torch.ones_like(B), D=None, z=None, delta_softplus=True) # TODO: gating where

        # memory = y_local[:, :, ::self.step]
        # att = self.w_O(self.global_attention(x, rearrange(memory, "b d m -> b m d")))
        # att_gated = att * rearrange(E, "b d l -> b l d")

        # return rearrange(y, "b d l -> b l d") + att_gated + x * rearrange(F, "b d l -> b l d")
        return rearrange(y, "b d l -> b l d")


    def global_attention(self, x: Tensor, memory: Tensor):
        Q: Tensor = self.w_Q(x)
        K: Tensor = self.w_K(memory)
        O_w = F.softmax((Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor([self.d_hidden]).cuda()), dim=-1) # multiply by the values and weighted the sum!!!!!! poggers in the chat
        # print(O_w.size(), self.w_V(memory).size())
        return torch.einsum("b o l, b l n -> b o n", O_w, self.w_V(memory))



class MambaBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_ssm_hidden: int, dt_rank: int, n_conv: int = 4):
        super().__init__()

        # self.conv = nn.Conv1d(d_hidden, d_hidden, bias=True, kernel_size=n_conv, groups=d_in, padding=n_conv-3) # convolutions must be much longer to understand gene level relationships
        self.conv_weight = nn.Parameter(torch.empty(d_hidden, n_conv))
        self.conv_bias = nn.Parameter(torch.empty(d_hidden))

        k = math.sqrt(1/(d_hidden * n_conv))
        torch.nn.init.uniform_(self.conv_weight, -k, k)
        torch.nn.init.uniform_(self.conv_bias, -k, k)

        self.d_hidden = d_hidden

        # TODO: ADD GATING TO the CONVOLUTION
        self.x_proj = nn.Linear(d_in, 2 * d_hidden)
        self.norm = nn.LayerNorm(d_in)

        self.x_res_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_in)

        self.ssm = S4Checkpointed(d_hidden, d_ssm_hidden, dt_rank, d_ssm_hidden)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        xz = self.x_proj(x_norm)
        x_in, z = xz.split([self.d_hidden, self.d_hidden], dim=-1)
        x_in = rearrange(x_in, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")
        main_x = rearrange(causal_conv1d_fn(x_in, self.conv_weight, self.conv_bias, activation="silu"), "b d l -> b l d")
        o = self.ssm(main_x, z)
        o = self.out_proj(o)

        return o + x


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.d_memory = 64

        self.n_heads = 8

        self.w_K = nn.Linear(self.d_inner, self.d_memory * self.n_heads, bias=False)
        self.w_Q = nn.Linear(self.d_inner, self.d_memory * self.n_heads, bias=False)
        self.w_V = nn.Linear(self.d_inner, self.d_memory * self.n_heads, bias=False)
        self.w_O = nn.Linear(self.d_memory * self.n_heads, self.d_inner)
        self.w_mlp = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner * 4, bias=bias),
            nn.SiLU(),
            nn.Linear(self.d_inner * 4, self.d_inner, bias=bias),
            RMSNorm(self.d_inner)
        )

        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_normal_(self.w_Q.weight)
        nn.init.xavier_normal_(self.w_K.weight)
        nn.init.xavier_normal_(self.w_V.weight)
        nn.init.xavier_normal_(self.w_O.weight)

        self.step = 512

        self.norm1 = RMSNorm(self.d_inner)


    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        # if inference_params is not None:
        #     conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         # The states are updated inplace
        #         out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        #         return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)

        y = rearrange(y, "b d l -> b l d")
        memory = y[:, ::self.step, :]
        att = self.w_O(self.global_attention(y, memory))
        # print(att.size(), rearrange(E, "(b l) dstate -> b l dstate", l=seqlen).contiguous().size())
        # att_gated = att * rearrange(E, "(b l) dstate -> b l dstate", l=seqlen).contiguous() TEST USING GATED ATTENTION

        out = self.w_mlp(self.norm1(y + att))

        # return self.out_proj(out + rearrange(x, "b d l -> b l d"))
        return self.out_proj(out)
        # return self.out_proj(y)

    def global_attention(self, x: Tensor, memory: Tensor):
        b, l, d = x.size()
        _b, lm, _d = memory.size()
        Q: Tensor = self.w_Q(x).view(b, self.n_heads, l, self.d_memory)
        K: Tensor = self.w_K(memory).view(b, self.n_heads, lm, self.d_memory)
        O_w = F.scaled_dot_product_attention(Q, K, self.w_V(memory).view(b, self.n_heads, lm, self.d_memory)).permute(0, 2, 1, 3).flatten(-2)
        return O_w
        # O_w = F.softmax((Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor([self.d_inner]).cuda()), dim=-1) # multiply by the values and weighted the sum!!!!!! poggers in the chat
        # print(O_w.size(), self.w_V(memory).size())
        # return torch.einsum("b o l, b l n -> b o n", O_w, self.w_V(memory))

# recurrent implementation (will use later for inference optimization)
        # for i in range(l):
        #     h = Ad[:, i] * h + Bx[:, i]
        #     h_local = A_localx[:, i] * h_local + Bx[:, i]
        #     if i % self.step == 0: memory[:, i // self.step] = h_local.flatten(start_dim=-2, end_dim=-1)
        #     att = self.global_attention(x[:, i].unsqueeze(0), memory, (i // self.step) + 1)
        #     att = self.w_O(att)
        #     att = self.w_E(att).reshape(b, self.d_in)
        #     y = torch.bmm(h, C[:, i, :].unsqueeze(-1)).reshape(b, self.d_in) + att # @ E[:, i, :] TODO: ADD IN THIS E MATRIX LATER
        #     ys.append(y)
        # y = torch.stack(ys, dim=1) + x * self.D
