import torch
import torch.nn as nn

class SSN_CNN_Solver(nn.Module):
    def __init__(self, mid_channels=1, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, f_z):
        out = f_z.unsqueeze(0).unsqueeze(0)
        out = self.net(out)
        return -out.squeeze(0).squeeze(0)


class J_update(nn.Module):
    def __init__(self, mid_channels=1, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=kernel_size, padding=padding),
        )
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, m):
        out = m.unsqueeze(0).unsqueeze(0)
        out = self.net(out)
        delta = out.squeeze(0).squeeze(0)
        return delta
    
class E_update(nn.Module):
    def __init__(self, mid_channels=1, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=kernel_size, padding=padding),
        )
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, m):
        out = m.unsqueeze(0).unsqueeze(0)
        out = self.net(out)
        delta = out.squeeze(0).squeeze(0)
        return delta                 # no ReLU


class Learned_SSN_Stage(nn.Module):
    def __init__(self, d, kernel_size=7):
        super().__init__()
        self.j_prox = J_update(kernel_size=kernel_size)
        self.cnn_prox_E = E_update(kernel_size=kernel_size)
        self.solver = SSN_CNN_Solver(kernel_size=kernel_size)

        self.beta = nn.Parameter(torch.tensor(0.1))
        self.eta = nn.Parameter(torch.tensor(0.01))
        self.j_relax = nn.Parameter(torch.tensor(-4.0))

    def forward(self, X, A, C, E, Y1, Y2, J):
        s_beta = torch.abs(self.beta) + 1e-4

        m = C + Y1 / s_beta
        J_prox = self.j_prox(m)
        relax = torch.sigmoid(self.j_relax)
        J_next = (1.0 - relax) * J + relax * J_prox

        AC = A @ C
        E = self.cnn_prox_E(X - AC + Y2 / s_beta)

        res_J = Y1 + s_beta * (C - J_next)
        res_E = A.t() @ (Y2 + s_beta * (X - AC - E))
        F_z = res_J - res_E

        delta_C = self.solver(F_z)
        C_next = C + self.eta * delta_C

        Y1_next = Y1 + s_beta * (C_next - J_next)
        Y2_next = Y2 + s_beta * (X - A @ C_next - E)

        return C_next, E, Y1_next, Y2_next, F_z, J_next


class DeepUnfoldingSSN(nn.Module):
    def __init__(self, d, n, num_stages=10, kernel_size=7):
        super().__init__()
        self.d = d
        self.n = n
        self.stages = nn.ModuleList([Learned_SSN_Stage(d, kernel_size=kernel_size) for _ in range(num_stages)])

    def forward(self, X):
        device = X.device
        dtype = X.dtype
        n = self.n

        # Use raw data as dictionary.
        A = X

        C = torch.zeros((n, n), device=device, dtype=dtype)
        E = torch.zeros((self.d, n), device=device, dtype=dtype)
        Y1 = torch.zeros((n, n), device=device, dtype=dtype)
        Y2 = torch.zeros((self.d, n), device=device, dtype=dtype)
        J = torch.zeros((n, n), device=device, dtype=dtype)

        all_FZ = []
        all_J = []
        all_Z = []
        for stage in self.stages:
            C, E, Y1, Y2, F_Z, J = stage(X, A, C, E, Y1, Y2, J)
            all_FZ.append(F_Z)
            all_J.append(J * (1 - torch.eye(n, device=device, dtype=dtype)))
            all_Z.append(C * (1 - torch.eye(n, device=device, dtype=dtype)))

        Z_full = C * (1 - torch.eye(n, device=device, dtype=dtype))
        J_full = J * (1 - torch.eye(n, device=device, dtype=dtype))

        return Z_full, E, all_FZ, J_full, all_J, all_Z