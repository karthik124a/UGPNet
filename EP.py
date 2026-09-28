import torch
import torch.nn as nn


class EP:
    def __init__(self, H, y, sigma2, user_num, constellation, batch_size):
        self.H = H
        self.y = y
        self.sigma2 = sigma2
        self.user_num = user_num
        self.batch_size = batch_size
        self.constellation = constellation
        self.soft_max = nn.Softmax(dim=2)

        self.constellation_expanded = constellation.tile(batch_size, 1).unsqueeze(2)
        self.constellation_expanded_transpose = constellation.tile(batch_size, user_num, 1)

    def calculate_mean_var(self, pyx, alpha, beta):
        mean = torch.matmul(pyx, self.constellation_expanded) * alpha
        var = torch.square(torch.abs(self.constellation_expanded_transpose - mean))
        var = torch.mul(pyx, var)
        var = torch.sum(var, axis=2) * beta
        return torch.squeeze(mean), var

    def LMMSE(self, diag_lamda, H, y, sigma2, lamda, gamma):
        # H is rectangular for MIMO: [batch, Nr_real, Nt_real].
        HtH = torch.matmul(H.permute(0, 2, 1), H)
        Hty = torch.squeeze(
            torch.matmul(H.permute(0, 2, 1), torch.unsqueeze(y, 2)), dim=2
        )

        sigma2_b = sigma2.reshape(-1, 1, 1)
        torch.einsum("ijj->ij", diag_lamda)[...] = lamda * sigma2_b.squeeze(-1)

        var = torch.linalg.inv(HtH + diag_lamda)
        mean = torch.matmul(
            var, torch.unsqueeze(Hty + gamma * sigma2_b.squeeze(-1), 2)
        )
        var = var * sigma2_b
        return mean, var

    def performEP(
        self, eta, diag_lamda, p_y_x_GNN, mean_ab_prev, var_ab_prev,
        lamda_prev, gamma_prev, iter_num, alpha, beta, k, l
    ):
        if iter_num == 0:
            lamda = lamda_prev.squeeze()
            gamma = gamma_prev.squeeze()
        else:
            p_y_x_GNN = self.soft_max(p_y_x_GNN)
            mean_b, var_b = self.calculate_mean_var(p_y_x_GNN, alpha, beta)
            var_b = torch.clamp(var_b, 1e-13, None)

            lamda_new = k * (1 / var_b - 1 / var_ab_prev)
            gamma_new = l * (mean_b / var_b - mean_ab_prev / var_ab_prev)

            bad = lamda_new < 0
            lamda_new = torch.where(bad, lamda_prev, lamda_new)
            gamma_new = torch.where(bad, gamma_prev, gamma_new)

            lamda = eta * lamda_prev + (1 - eta) * lamda_new
            gamma = eta * gamma_prev + (1 - eta) * gamma_new

        mean_mmse, var_mmse = self.LMMSE(
            diag_lamda, self.H, self.y, self.sigma2, lamda, gamma
        )

        diag_var = torch.diagonal(var_mmse, dim1=1, dim2=2)
        var_ab = 1 / (1 / diag_var - lamda)
        var_ab = torch.clamp(var_ab, 1e-13, None)

        mean_ab = torch.squeeze(mean_mmse) / diag_var - gamma
        mean_ab = var_ab * mean_ab

        return mean_ab, var_ab, lamda, gamma
