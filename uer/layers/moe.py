import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, has_bias=True):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        # 定义多个专家，每个专家是一个小型 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=has_bias),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim, bias=has_bias)
            ) for _ in range(num_experts)
        ])
        # 门控网络，用于选择专家
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        gate_scores = torch.softmax(self.gate(x), dim=-1)  # [batch_size, seq_length, num_experts]
        # 计算每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [batch_size, seq_length, input_dim, num_experts]
        # 加权组合专家输出
        output = torch.einsum('bsed,bsd->bse', expert_outputs, gate_scores)  # [batch_size, seq_length, input_dim]
        return output, gate_scores  # 返回输出和门控分数