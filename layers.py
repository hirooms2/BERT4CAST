import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(AdditiveAttention, self).__init__()
        self.affine1 = nn.Linear(in_features=feature_dim, out_features=attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=attention_dim, out_features=1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))  # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)  # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)  # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)  # [batch_size, feature_dim]
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        #       [bz, 20, seq_len, 20] x [bz, 20, 20, seq_len] -> [bz, 20, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        #       [bz, 20, seq_len, seq_len] x [bz, 20, seq_len, 20] -> [bz, 20, seq_len, 20]
        context = torch.matmul(attn, V)
        return context, attn


class Conv1D(nn.Module):
    def __init__(self, cnn_method: str, in_channels: int, cnn_kernel_num: int, cnn_window_size: int):
        super(Conv1D, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group5']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        if self.cnn_method == 'naive':
            self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num,
                                  kernel_size=cnn_window_size, padding=(cnn_window_size - 1) // 2)
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=1,
                                   padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=3,
                                   padding=1)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=5,
                                   padding=2)
        else:
            assert cnn_kernel_num % 5 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=1,
                                   padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=2,
                                   padding=0)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=3,
                                   padding=1)
            self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=4,
                                   padding=1)
            self.conv5 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=5,
                                   padding=2)
        self.device = torch.device('cuda')

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num, length]
    def forward(self, feature):
        if self.cnn_method == 'naive':
            return F.relu(self.conv(feature))  # [batch_size, cnn_kernel_num, length]
        elif self.cnn_method == 'group3':
            return F.relu(torch.cat([self.conv1(feature), self.conv2(feature), self.conv3(feature)], dim=1))
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1], device=self.device)
            return F.relu(torch.cat([self.conv1(feature), \
                                     self.conv2(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv3(feature), \
                                     self.conv4(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv5(feature)], dim=1))


class Context_Aware_Att(nn.Module):
    def __init__(self, nb_head: int, size_per_head: int, d_model: int, len_q: int, len_k: int):
        super(Context_Aware_Att, self).__init__()

        self.n_heads = nb_head
        self.d_k = size_per_head
        self.hidden_size = nb_head * size_per_head

        self.len_q = len_q
        self.len_k = len_k

        self.attention_scalar = math.sqrt(float(self.d_k))

        self.W_Q = nn.Linear(in_features=d_model, out_features=self.hidden_size)
        self.W_K = nn.Linear(in_features=d_model, out_features=self.hidden_size)
        self.W_V = nn.Linear(in_features=d_model, out_features=self.hidden_size)
        # self.W_P = nn.Linear(in_features=self.hidden_size, out_features=1)

        # self.F1 = nn.Linear(self.hidden_size, 1024, bias=True)
        # self.F2 = nn.Linear(1024, self.hidden_size, bias=True)
        # self.layernorm = nn.LayerNorm(self.hidden_size)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)

    # Input
    # Q    : [batch_size, len_q, dim]
    # K    : [batch_size, len_k, dim]
    # V    : [batch_size, len_k, dim]
    # title_mask : [batch_size, len_q]
    # body_mask : [batch_size, len_k]
    # Output
    # out  : [batch_size, len_q, nb_head * size_per_head]
    def forward(self, Q_seq, K_seq, V_seq, title_mask, body_mask):
        batch_size = Q_seq.size(0)
        title_len = Q_seq.size(1)
        body_len = Q_seq.size(2)

        K_seq = torch.cat([Q_seq, K_seq], 1)  # [B, M, d] -> [B, N+M, d] Body, Title
        V_seq = torch.cat([Q_seq, V_seq], 1)  # [B, M, d] -> [B, N+M, d]

        # mask = torch.cat([body_mask, title_mask], dim=1)  # [B, N+M]
        mask = title_mask.unsqueeze(1).repeat(1, self.len_q, 1)  # [bz, N, N]
        body_mask = body_mask.unsqueeze(1).repeat(1, self.len_q, 1)  # [B, N, M]
        mask = torch.cat([mask, body_mask], dim=2)  # [B, N, N+M]
        mask = mask * title_mask.unsqueeze(-1)  # [B, N, N+M]
        mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [bz, 20, seq_len, seq_len]

        Q = self.W_Q(Q_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B, nh, N, nd]
        K = self.W_K(K_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B, nh, N+M, nd]
        V = self.W_V(V_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B, nh, N+M, nd]

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # [B, nh, N, N+M]
        attention = F.softmax(logits.masked_fill(mask == 0, -1e9), dim=3)  # [B, nh, N, N+M]

        # hidden = torch.matmul(attn, V)  # [B, nh, N, nd]
        # hidden = hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)  # [bz, seq_len, 400]
        # hidden = Q_seq + hidden

        hidden = torch.matmul(attention, V)  # [B, nh, N, nd]
        hidden = hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)  # [bz, seq_len, 400]
        # # # Drop-out -> Add & norm
        # new_hidden = F.dropout(new_hidden, p=0.1, training=self.training)
        # hidden = self.layernorm(Q_seq + new_hidden)
        #
        # # Point-wise Feed-forward
        # new_hidden = self.F2(torch.nn.GELU()(self.F1(hidden)))
        # new_hidden = F.dropout(new_hidden, p=0.1, training=self.training)
        # hidden = self.layernorm(hidden + new_hidden)

        return hidden


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, enable_gpu=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        self.enable_gpu = enable_gpu

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        batch_size, seq_len, _ = Q.shape

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)  # [bz, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [bz, 20, seq_len, seq_len]

        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)  # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  # self.layer_norm(output + residual)


class WeightedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(WeightedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_softmax = nn.Softmax(dim=-1)(self.weight)
        return F.linear(input, weight_softmax)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
