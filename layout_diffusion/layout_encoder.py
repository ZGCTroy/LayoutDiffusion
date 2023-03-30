"""
Transformer implementation adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py
"""

import math

import torch
import torch as th
import torch.nn as nn


def xf_convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x, key_padding_mask=None):
        x = self.c_qkv(x)
        x = self.attention(x, key_padding_mask)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv, key_padding_mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        weight = th.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards

        if key_padding_mask is not None:
            weight = weight.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, L1)
                float('-inf'),
            )
        wdtype = weight.dtype
        weight = th.softmax(weight.float(), dim=-1).type(wdtype)
        return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: th.Tensor, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: th.Tensor, key_padding_mask=None):
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        return x


class LayoutTransformerEncoder(nn.Module):
    def __init__(
            self,
            layout_length: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            num_heads: int,
            use_final_ln: bool,
            num_classes_for_layout_object: int,
            mask_size_for_layout_object: int,
            used_condition_types=['obj_class', 'obj_bbox', 'obj_mask'],
            use_positional_embedding=True,
            resolution_to_attention=[],
            use_key_padding_mask=False,
            not_use_layout_fusion_module=False
    ):
        super().__init__()
        self.not_use_layout_fusion_module=not_use_layout_fusion_module
        self.use_key_padding_mask = use_key_padding_mask
        self.used_condition_types = used_condition_types
        self.num_classes_for_layout_object = num_classes_for_layout_object
        self.mask_size_for_layout_object = mask_size_for_layout_object
        if not self.not_use_layout_fusion_module:
            self.transform = Transformer(
                n_ctx=layout_length,
                width=hidden_dim,
                layers=num_layers,
                heads=num_heads
            )
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(th.empty(layout_length, hidden_dim, dtype=th.float32))
        self.transformer_proj = nn.Linear(hidden_dim, output_dim)

        if 'obj_class' in self.used_condition_types:
            self.obj_class_embedding = nn.Embedding(num_classes_for_layout_object, hidden_dim)
        if 'obj_bbox' in self.used_condition_types:
            self.obj_bbox_embedding = nn.Linear(4, hidden_dim)
        if 'obj_mask' in self.used_condition_types:
            self.obj_mask_embedding = nn.Linear(mask_size_for_layout_object * mask_size_for_layout_object, hidden_dim)

        if use_final_ln:
            self.final_ln = LayerNorm(hidden_dim)
        else:
            self.final_ln = None

        self.dtype = torch.float32

        self.resolution_to_attention = resolution_to_attention
        self.image_patch_bbox_embedding = {}
        for resolution in self.resolution_to_attention:
            interval = 1.0 / resolution
            self.image_patch_bbox_embedding['resolution{}'.format(resolution)] = torch.FloatTensor(
                [(interval * j, interval * i, interval * (j + 1), interval * (i + 1)) for i in range(resolution) for j in range(resolution)],
            ).cuda()  # (L, 4)

    def convert_to_fp16(self):
        self.dtype = torch.float16
        if not self.not_use_layout_fusion_module:
            self.transform.apply(xf_convert_module_to_f16)
        self.transformer_proj.to(th.float16)
        if self.use_positional_embedding:
            self.positional_embedding.to(th.float16)
        if 'obj_class' in self.used_condition_types:
            self.obj_class_embedding.to(th.float16)
        if 'obj_bbox' in self.used_condition_types:
            self.obj_bbox_embedding.to(th.float16)
        if 'obj_mask' in self.used_condition_types:
            self.obj_mask_embedding.to(th.float16)

    def forward(self, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, image_patch_bbox=None):
        assert (obj_class is not None) or (obj_bbox is not None) or (obj_mask is not None)
        outputs = {}

        xf_in = None
        if self.use_positional_embedding:
            xf_in = self.positional_embedding[None]

        if 'obj_class' in self.used_condition_types:
            obj_class_embedding = self.obj_class_embedding(obj_class.long())
            if xf_in is None:
                xf_in = obj_class_embedding
            else:
                xf_in = xf_in + obj_class_embedding
            outputs['obj_class_embedding'] = obj_class_embedding.permute(0, 2, 1)

        if 'obj_bbox' in self.used_condition_types:
            obj_bbox_embedding = self.obj_bbox_embedding(obj_bbox.to(self.dtype))
            if xf_in is None:
                xf_in = obj_bbox_embedding
            else:
                xf_in = xf_in + obj_bbox_embedding
            outputs['obj_bbox_embedding'] = obj_bbox_embedding.permute(0, 2, 1)
            for resolution in self.resolution_to_attention:
                outputs['image_patch_bbox_embedding_for_resolution{}'.format(resolution)] = torch.repeat_interleave(
                    input=self.obj_bbox_embedding(
                        self.image_patch_bbox_embedding['resolution{}'.format(resolution)].to(self.dtype)
                    ).unsqueeze(0),
                    repeats = obj_bbox_embedding.shape[0],
                    dim=0
                ).permute(0, 2, 1)

        if 'obj_mask' in self.used_condition_types:
            if xf_in is None:
                xf_in = self.obj_mask_embedding(obj_mask.view(*obj_mask.shape[:2], -1).to(self.dtype))
            else:
                xf_in = xf_in + self.obj_mask_embedding(obj_mask.view(*obj_mask.shape[:2], -1).to(self.dtype))

        if 'is_valid_obj' in self.used_condition_types:
            outputs['key_padding_mask'] = (1-is_valid_obj).bool() # (N, L2)

        key_padding_mask = outputs['key_padding_mask'] if self.use_key_padding_mask else None
        if self.not_use_layout_fusion_module:
            xf_out = xf_in.to(self.dtype)
        else:
            xf_out = self.transform(xf_in.to(self.dtype), key_padding_mask)  # NLC

        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, 0])  # NC
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs['xf_proj'] = xf_proj
        outputs['xf_out'] = xf_out

        return outputs
