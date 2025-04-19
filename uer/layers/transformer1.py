# uer/layers/transformer.py（修改后）
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.moe import MoELayer  # 导入 MoE 层

class TransformerLayer(nn.Module):
    def __init__(self, args):
        super(TransformerLayer, self).__init__()
        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout,
            has_bias=has_bias, with_scale=with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # 使用 MoE 层替换原有的 FFN
        self.feed_forward = MoELayer(
            input_dim=args.hidden_size,
            hidden_dim=args.feedforward_size,
            num_experts=args.num_experts,  # 新增参数，需在 args 中定义
            has_bias=has_bias
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)

    def forward(self, hidden, mask, position_bias=None):
        if self.layernorm_positioning == "post":
            inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask, position_bias))
            inter = self.layer_norm_1(inter + hidden)
            output, gate_scores = self.feed_forward(inter)  # 接收 MoELayer 返回的 (output, gate_scores)
            output = self.dropout_2(output)
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter = self.dropout_1(self.self_attn(inter, inter, inter, mask, position_bias))
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output, gate_scores = self.feed_forward(output)  # 接收 MoELayer 返回的 (output, gate_scores)
            output = self.dropout_2(output) + hidden
        return output, gate_scores  # 返回 (output, gate_scores)