import torch
import torch.nn as nn

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB_Prox_Net(nn.Module):
    """
    使用残差密集块 (RDB) 替代原有的 CNN_Prox_Net
    针对矩阵结构学习更强的低秩/稀疏先验
    """
    def __init__(self, growRate0=32, growRate=16, nConvLayers=3, lambd=0.01):
        super(RDB_Prox_Net, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        # 1. 输入升维层
        self.in_conv = nn.Conv2d(1, G0, kernel_size=3, padding=1)
        
        # 2. 密集卷积层 (Dense Layers)
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        
        # 3. 局部特征融合 (Local Feature Fusion)
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0)
        
        # 4. 输出降维层
        self.out_conv = nn.Conv2d(G0, 1, kernel_size=3, padding=1)
        
        # 5. 稀疏与非负算子
        self.shrink = nn.Softshrink(lambd=lambd)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (d, n) -> (1, 1, d, n)
        identity = x.unsqueeze(0).unsqueeze(0)
        
        # RDB 主体结构
        out = self.in_conv(identity)
        f_dense = self.convs(out)
        out = self.LFF(f_dense) + out  # 局部残差学习
        out = self.out_conv(out)
        
        out = out.squeeze(0).squeeze(0)
        return self.relu(out)  # 强制非负
    



import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeInit(nn.Module):
    """h0 = phi(X): X:(d,N) -> h0:(N,hidden) 只算一次"""
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

    def forward(self, X):  # X:(d,N)
        return self.net(X.t())  # (N,hidden)


class EdgeCondResMPLayer(nn.Module):
    """
    Residual message passing on kNN graph, edge feature is m_ij (from M).
    用线性分解避免大 concat：ReLU(Wi hi + Wj hj + Wm m_ij)
    """
    def __init__(self, hidden=64):
        super().__init__()
        # attention
        self.Wi = nn.Linear(hidden, hidden, bias=False)
        self.Wj = nn.Linear(hidden, hidden, bias=False)
        self.Wm_att = nn.Linear(1, hidden, bias=False)
        self.att_out = nn.Linear(hidden, 1, bias=False)

        # message
        self.Wmsg_h = nn.Linear(hidden, hidden, bias=False)
        self.Wm_msg = nn.Linear(1, hidden, bias=False)

        # update
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, nbr_idx, m_e):
        """
        h: (N,hidden)
        nbr_idx: (N,k) long
        m_e: (N,k) edge values sampled from M
        """
        N, k = nbr_idx.shape

        hi = h                                    # (N,h)
        hj = h[nbr_idx]                            # (N,k,h)

        hi_att = self.Wi(hi).unsqueeze(1)          # (N,1,h)
        hj_att = self.Wj(hj)                       # (N,k,h)
        m_att  = self.Wm_att(m_e.unsqueeze(-1))    # (N,k,h)

        a = F.relu(hi_att + hj_att + m_att)        # (N,k,h)
        logits = self.att_out(a).squeeze(-1)       # (N,k)
        alpha = F.softmax(logits, dim=1)           # over k neighbors

        msg = F.relu(self.Wmsg_h(hj) + self.Wm_msg(m_e.unsqueeze(-1)))  # (N,k,h)
        agg = torch.sum(alpha.unsqueeze(-1) * msg, dim=1)              # (N,h)

        dh = self.upd(torch.cat([h, agg], dim=-1))  # (N,h)
        return h + dh                               # residual


class JProx_GNN_OnceX(nn.Module):
    """
    输入：h0(固定,来自X一次初始化), 当前 M, nbr_idx
    输出：J = M + Delta (dense NxN)
    """
    def __init__(self, n, hidden=64, mp_layers=2, nonneg_J=True, zero_diag=True, symmetrize=False):
        super().__init__()
        self.n = n
        self.hidden = hidden
        self.mp = nn.ModuleList([EdgeCondResMPLayer(hidden) for _ in range(mp_layers)])

        # dense all-pairs decoder
        self.Wq = nn.Linear(hidden, hidden, bias=False)
        self.Wk = nn.Linear(hidden, hidden, bias=False)

        # 稳定训练：初始 Delta≈0，使第一步 J≈M
        self.delta_scale = nn.Parameter(torch.tensor(0.0))

        self.nonneg_J = nonneg_J
        self.zero_diag = zero_diag
        self.symmetrize = symmetrize
        self.register_buffer("off_diag_mask", (1 - torch.eye(n)))

    def forward(self, h0, M, nbr_idx):
        """
        h0: (N,hidden) from X (computed once)
        M:  (N,N)
        nbr_idx: (N,k)
        """
        N = M.shape[0]
        assert N == self.n

        # edge feature from current M on kNN edges
        row = torch.arange(N, device=M.device).unsqueeze(1).expand_as(nbr_idx)
        m_e = M[row, nbr_idx]  # (N,k)

        # message passing driven by M (h starts from fixed h0)
        h = h0
        for layer in self.mp:
            h = layer(h, nbr_idx, m_e)

        # dense decode to Delta
        Q = self.Wq(h)
        K = self.Wk(h)
        Delta = (Q @ K.t()) / math.sqrt(self.hidden)          # (N,N)
        Delta = torch.tanh(self.delta_scale) * Delta          # keep small early

        if self.symmetrize:
            Delta = 0.5 * (Delta + Delta.t())

        if self.zero_diag:
            Delta = Delta * self.off_diag_mask.to(dtype=Delta.dtype)

        J = M + Delta

        if self.nonneg_J:
            J = F.relu(J)
        if self.zero_diag:
            J = J * self.off_diag_mask.to(dtype=J.dtype)

        return J
