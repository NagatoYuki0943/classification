# https://github.com/sooftware/conformer

# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(
            2, dim=self.dim
        )  # [B, C*2, N] -> [B, C, N], [B, C, N]
        return outputs * gate.sigmoid()  # [B, C, N] * sigmoid([B, C, N]) = [B, C, N]


class Conv2dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            ),  # add padding
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            ),  # add padding
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(
            inputs.unsqueeze(1)
        )  # [B, N, 80] -> [B, 512, N/4, 20]
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)  # [B, 512, N/4, 20] -> [B, N/4, 512, 20]
        outputs = outputs.contiguous().view(
            batch_size, subsampled_lengths, channels * sumsampled_dim
        )  # [B, N/4, 512, 20] -> [B, N/4, 512*20]

        output_lengths = input_lengths >> 2  # input_lengths // 4
        # output_lengths -= 1   # add padding

        return outputs, output_lengths


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """

    def __init__(
        self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0
    ):
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


# ---------------------#
#   ConvModule
#
#       │
#       ├─────┐
#       │     │
#      ln     │
#       │     │
#     pwconv  │
#       │     │
#      glu    │
#       │     │
#     dwconv  │
#       │     │
#      bn     │
#       │     │
#      silu   │
#       │     │
#     pwconv  │
#       │     │
#     drop    │
#       │     │
#       + ────┘
#       │
# ---------------------#
class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.ln = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * expansion_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.glu = GLU(dim=1)

        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(in_channels)
        self.silu = nn.SiLU()

        self.conv3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.ln(inputs)
        x = x.transpose(1, 2)  # [B, N, C] -> [B, C, N]
        x = self.conv1(x)  # [B, C, N] -> [B, C*2, N]
        x = self.glu(x)  # [B, C*2, N] -> [B, C, N]
        x = self.conv2(x)  # [B, C, N] -> [B, C, N]
        x = self.bn(x)
        x = self.silu(x)
        x = self.conv3(x)  # [B, C, N] -> [B, C, N]
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, C, N] -> [B, N, C]
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """

    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query)  # [B, N, C] -> [B, N, C]
        query = query.view(
            batch_size, -1, self.num_heads, self.d_head
        )  # [B, N, C] -> [B, N, num_head, head_dim]   C = num_head * head_dim

        key = self.key_proj(key)  # [B, N, C] -> [B, N, C]
        key = key.view(
            batch_size, -1, self.num_heads, self.d_head
        )  # [B, N, C] -> [B, N, num_head, head_dim]
        key = key.permute(
            0, 2, 1, 3
        )  # [B, N, num_head, head_dim] -> [B, num_head, N, head_dim]

        value = self.value_proj(value)  # [B, N, C] -> [B, N, C]
        value = value.view(
            batch_size, -1, self.num_heads, self.d_head
        )  # [B, N, C] -> [B, N, num_head, head_dim]
        value = value.permute(
            0, 2, 1, 3
        )  # [B, N, num_head, head_dim] -> [B, num_head, N, head_dim]

        pos_embedding = self.pos_proj(pos_embedding)  # [B, N, C] -> [B, N, C]
        pos_embedding = pos_embedding.view(
            batch_size, -1, self.num_heads, self.d_head
        )  # [B, N, C] -> [B, N, num_head, head_dim]

        content_score = torch.matmul(
            (query + self.u_bias).transpose(1, 2), key.transpose(2, 3)
        )  # [B, num_head, N, head_dim] @ [B, num_head, head_dim, N] = [B, num_head, N, N]
        pos_score = torch.matmul(
            (query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1)
        )  # [B, num_head, N, head_dim] @ [B, num_head, head_dim, N] = [B, num_head, N, N]
        pos_score = self._relative_shift(
            pos_score
        )  # [B, num_head, N, N] -> [B, num_head, N, N]

        score = (
            (content_score + pos_score) / self.sqrt_dim
        )  # ([B, num_head, N, N] + [B, num_head, N, N]) / sqrt_dim = [B, num_head, N, N]

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(
            attn, value
        ).transpose(
            1, 2
        )  # [B, num_head, N, N] @ [B, num_head, N, head_dim] = [B, num_head, N, head_dim] -> [B, N, num_head, head_dim]
        context = context.contiguous().view(
            batch_size, -1, self.d_model
        )  # [B, N, num_head, head_dim] -> [B, N, C]

        return self.out_proj(context)  # [B, N, C] -> [B, N, C]

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1
        )
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()  # [B, N, C]
        pos_embedding = self.positional_encoding(seq_length)  # [1, N, C]
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)  # [1, N, C] -> [B, N, C]

        inputs = self.layer_norm(inputs)
        outputs = self.attention(
            inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask
        )  # [B, N, C] -> [B, N, C]

        return self.dropout(outputs)


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(encoder_dim)
        self.fc1 = Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(p=dropout_p)
        self.fc2 = Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.drop2 = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.norm(inputs)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


# ---------------------#
# 单个ConformerBlock
#
#       │
#       ├─────────┐
#       │         │
#      mlp        │
#       │         │
#       + ────────┘
#       │
#       ├─────────┐
#       │         │
#     attn        │
#       │         │
#       + ────────┘
#       │
#       ├─────────┐
#       │         │
#   ConvModule    │
#       │         │
#       + ────────┘
#       │
#       ├─────────┐
#       │         │
#      mlp        │
#       │         │
#       + ────────┘
#       │
#      ln
#       │
# ---------------------#
class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,  # 隐藏层维度
        num_attention_heads: int = 8,  # attnention头数
        feed_forward_expansion_factor: int = 4,  # mlp扩展系数
        conv_expansion_factor: int = 2,  # 卷积扩展系数
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super().__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.ffn1 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )
        self.attn = ResidualConnectionModule(
            module=MultiHeadedSelfAttentionModule(
                d_model=encoder_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p,
            ),
        )
        self.conv = ResidualConnectionModule(
            module=ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
            ),
        )
        self.ffn2 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )
        self.ln = nn.LayerNorm(encoder_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.ffn1(inputs)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.ln(x)
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer encoder first processes the input with a convolution subsampling layer and then
    with a number of conformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        input_dim: int = 80,  # 输入数据的维度
        encoder_dim: int = 512,  # 隐藏层维度
        num_layers: int = 17,  # 重复block次数
        num_attention_heads: int = 8,  # attnention头数
        feed_forward_expansion_factor: int = 4,  # mlp扩展系数
        conv_expansion_factor: int = 2,  # 卷积扩展系数
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,  # 深度卷积核大小
        half_step_residual: bool = True,
    ):
        super().__init__()
        # 开始的下采样
        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)

        # 调整输入通道
        self.input_projection = nn.Sequential(
            # Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            Linear(
                encoder_dim * (input_dim // 2 // 2), encoder_dim
            ),  # Conv2dSubampling add padding
            nn.Dropout(p=input_dropout_p),
        )

        # 循环 ConformerBlock
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual,
                )
                for _ in range(num_layers)
            ]
        )

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """Update dropout probability of encoder"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        # 开始的下采样 [B, N, 80], [B] -> [B, N/4, 512*20], [B]
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)

        # 调整输入通道  [B, N/4, 512*20] -> [B, N/4, 512]
        outputs = self.input_projection(outputs)

        # 循环 ConformerBlock
        for layer in self.layers:
            outputs = layer(outputs)  # [B, N/4, 512] -> [B, N/4, 512]

        return outputs, output_lengths


class Conformer(nn.Module):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        num_classes: int,  # 分类数
        input_dim: int = 80,  # 输入数据的维度
        encoder_dim: int = 512,  # 隐藏层维度
        num_encoder_layers: int = 17,  # 重复block次数
        num_attention_heads: int = 8,  # attnention头数
        feed_forward_expansion_factor: int = 4,  # mlp扩展系数
        conv_expansion_factor: int = 2,  # 卷积扩展系数
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,  # 深度卷积核大小
        half_step_residual: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """Update dropout probability of model"""
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, encoder_output_lengths = self.encoder(
            inputs, input_lengths
        )  # [B, N, 80], [B] -> [B, N/4, 512], [B]
        outputs = self.fc(encoder_outputs)  # [B, N/4, 512] -> [B, N/4, num_classes]
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs, encoder_output_lengths


if __name__ == "__main__":
    batch_size, sequence_length, input_dim = 3, 1000, 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CTCLoss()

    inputs = torch.rand(batch_size, sequence_length, input_dim).to(device)
    input_lengths = torch.LongTensor([1000, 900, 800])
    targets = torch.LongTensor(
        [
            [1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0],
        ]
    ).to(device)
    target_lengths = torch.LongTensor([10, 10, 10])

    model = Conformer(
        num_classes=10,
        input_dim=input_dim,
        num_encoder_layers=2,
    ).to(device)

    # Forward propagate
    outputs, output_lengths = model(inputs, input_lengths)
    print(outputs.shape)  # [B, 2500, 10]
    print(output_lengths.shape)  # [B]
    print(output_lengths)  # [250, 225, 200]

    # Calculate CTC Loss
    loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
    print(loss)  # 252.6319
