# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# References:
#   https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/eval/segmentation_m2f/models/backbones/vit.py

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn


class Mlp(nn.Module):

  def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      drop: float = 0.0,
      bias: bool = True,
  ) -> None:
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    self.drop = nn.Dropout(drop)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


def make_2tuple(x):
  if isinstance(x, tuple):
    assert len(x) == 2
    return x

  assert isinstance(x, int)
  return (x, x)


class PatchEmbed(nn.Module):
  """2D image to patch embedding: (B,C,H,W) -> (B,N,D)

  Args:
      img_size: Image size.
      patch_size: Patch token size.
      in_chans: Number of input image channels.
      embed_dim: Number of linear projection output channels.
      norm_layer: Normalization layer.
  """

  def __init__(
      self,
      img_size: Union[int, Tuple[int, int]] = 224,
      patch_size: Union[int, Tuple[int, int]] = 16,
      in_chans: int = 3,
      embed_dim: int = 768,
      norm_layer: Optional[Callable] = None,
      flatten_embedding: bool = True,
  ) -> None:
    super().__init__()

    image_HW = make_2tuple(img_size)
    patch_HW = make_2tuple(patch_size)
    patch_grid_size = (
        image_HW[0] // patch_HW[0],
        image_HW[1] // patch_HW[1],
    )

    self.img_size = image_HW
    self.patch_size = patch_HW
    self.patches_resolution = patch_grid_size
    self.num_patches = patch_grid_size[0] * patch_grid_size[1]

    self.in_chans = in_chans
    self.embed_dim = embed_dim

    self.flatten_embedding = flatten_embedding

    self.proj = nn.Conv2d(
        in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
    )
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _, _, H, W = x.shape
    patch_H, patch_W = self.patch_size

    assert (
        H % patch_H == 0
    ), f"Input image height {H} is not a multiple of patch height {patch_H}"
    assert (
        W % patch_W == 0
    ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

    x = self.proj(x)  # B C H W
    H, W = x.size(2), x.size(3)
    x = x.flatten(2).transpose(1, 2)  # B HW C
    x = self.norm(x)
    if not self.flatten_embedding:
      x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
    return x

  def flops(self) -> float:
    Ho, Wo = self.patches_resolution
    flops = (
        Ho
        * Wo
        * self.embed_dim
        * self.in_chans
        * (self.patch_size[0] * self.patch_size[1])
    )
    if self.norm is not None:
      flops += Ho * Wo * self.embed_dim
    return flops


XFORMERS_AVAILABLE = False


class Attention(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int = 8,
      qkv_bias: bool = False,
      proj_bias: bool = True,
      attn_drop: float = 0.0,
      proj_drop: float = 0.0,
  ) -> None:
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim**-0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim, bias=proj_bias)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )

    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class MemEffAttention(Attention):

  def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
    if not XFORMERS_AVAILABLE:
      assert attn_bias is None, "xFormers is required for nested tensors usage"
      return super().forward(x)
    else:
      raise NotImplementedError("MemEffAttention do not support xFormer")
    # B, N, C = x.shape
    # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

    # q, k, v = unbind(qkv, 2)

    # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
    # x = x.reshape([B, N, C])

    # x = self.proj(x)
    # x = self.proj_drop(x)
    # return x


class LayerScale(nn.Module):

  def __init__(
      self,
      dim: int,
      init_values: Union[float, torch.Tensor] = 1e-5,
      inplace: bool = False,
  ) -> None:
    super().__init__()
    self.inplace = inplace
    self.gamma = nn.Parameter(init_values * torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
  if drop_prob == 0.0 or not training:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,) * (
      x.ndim - 1
  )  # work with diff dim tensors, not just 2D ConvNets
  random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
  if keep_prob > 0.0:
    random_tensor.div_(keep_prob)
  output = x * random_tensor
  return output


class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

  def __init__(self, drop_prob=None):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int,
      mlp_ratio: float = 4.0,
      qkv_bias: bool = False,
      proj_bias: bool = True,
      ffn_bias: bool = True,
      drop: float = 0.0,
      attn_drop: float = 0.0,
      init_values=None,
      drop_path: float = 0.0,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      attn_class: Callable[..., nn.Module] = Attention,
      ffn_layer: Callable[..., nn.Module] = Mlp,
  ) -> None:
    super().__init__()
    # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
    self.norm1 = norm_layer(dim)
    self.attn = attn_class(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        attn_drop=attn_drop,
        proj_drop=drop,
    )
    self.ls1 = (
        LayerScale(dim, init_values=init_values)
        if init_values
        else nn.Identity()
    )
    self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = ffn_layer(
        in_features=dim,
        hidden_features=mlp_hidden_dim,
        act_layer=act_layer,
        drop=drop,
        bias=ffn_bias,
    )
    self.ls2 = (
        LayerScale(dim, init_values=init_values)
        if init_values
        else nn.Identity()
    )
    self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    self.sample_drop_ratio = drop_path

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
      return self.ls1(self.attn(self.norm1(x)))

    def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
      return self.ls2(self.mlp(self.norm2(x)))

    if self.training and self.sample_drop_ratio > 0.1:
      # the overhead is compensated only for a drop path rate larger than 0.1
      x = drop_add_residual_stochastic_depth(
          x,
          residual_func=attn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
      x = drop_add_residual_stochastic_depth(
          x,
          residual_func=ffn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
    elif self.training and self.sample_drop_ratio > 0.0:
      x = x + self.drop_path1(attn_residual_func(x))
      x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
    else:
      x = x + attn_residual_func(x)
      x = x + ffn_residual_func(x)
    return x


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
  # 1) extract subset using permutation
  b, n, d = x.shape
  sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
  brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
  x_subset = x[brange]

  # 2) apply residual_func to get residual
  residual = residual_func(x_subset)

  x_flat = x.flatten(1)
  residual = residual.flatten(1)

  residual_scale_factor = b / sample_subset_size

  # 3) add the residual
  x_plus_residual = torch.index_add(
      x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
  )
  return x_plus_residual.view_as(x)
