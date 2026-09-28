import numpy as np
import torch


class genData(object):
    """Generate true MIMO-OTFS training/testing data.

    Nt/Nr are physical transmit/receive antenna counts.
    M/N are OTFS delay/time grid dimensions.

    The complex channel has shape:
        Hc: [Nr*M*N, Nt*M*N]
    and the real equivalent channel has shape:
        H:  [2*Nr*M*N, 2*Nt*M*N]
    """

    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.constellation = params['constellation']
        self.Nt = params['Nt']
        self.Nr = params['Nr']
        self.M = params['M']
        self.N = params['N']
        self.SNR_dB_min = params['SNR_dB_min']
        self.SNR_dB_max = params['SNR_dB_max']
        self.iter_EP_gD = params['iter_EP_gD']
        self.compare = params['compare']

        self.num_tx_symbols = self.Nt * self.M * self.N
        self.num_rx_symbols = self.Nr * self.M * self.N
        self.num_tx_real = 2 * self.num_tx_symbols
        self.num_rx_real = 2 * self.num_rx_symbols

        self.taps = 4
        self.delay_taps = torch.arange(self.taps, dtype=torch.long) % self.M
        self.doppler_taps = torch.arange(self.taps, dtype=torch.long) % self.N
        self.pow_prof = torch.ones(self.taps, dtype=torch.float64) / self.taps

    def dataTrain(self):
        # Each complex transmit symbol is represented by two real variables.
        s = np.random.randint(
            low=0,
            high=self.constellation.shape[0],
            size=[self.batch_size, self.num_tx_real]
        )
        self.s = s

        x_real = self.constellation[s].astype(np.float64)
        H_real = np.zeros(
            (self.batch_size, self.num_rx_real, self.num_tx_real),
            dtype=np.float64
        )

        # Generate an independent MIMO channel for every sample and antenna pair.
        for b in range(self.batch_size):
            Hc = self._generate_mimo_channel()
            H_real[b] = self._complex_to_real(Hc)

        # y = Hx + n
        y_real = np.squeeze(np.matmul(H_real, np.expand_dims(x_real, 2)), axis=2)

        snr_db = np.random.uniform(
            self.SNR_dB_min, self.SNR_dB_max, [self.batch_size, 1]
        )

        # Unit-average-noise convention. The 1/sqrt(Nt) channel scaling keeps
        # received signal power approximately stable as Nt changes.
        sigma2 = 1.0 / np.power(10.0, snr_db / 10.0)
        noise = np.sqrt(sigma2 / 2.0) * np.random.randn(
            self.batch_size, self.num_rx_real
        )
        y_noise_real = y_real + noise

        # Real-valued noise variance used by the real equivalent system.
        sigma2 = sigma2 / 2.0

        SER_mmse = None
        SER_EP = None
        SER_ML = None

        if self.compare:
            SER_mmse, _, _ = self.MMSE(
                x_real, y_noise_real, H_real, sigma2
            )
            SER_EP = self.EP(
                x_real, y_noise_real, H_real, sigma2, self.iter_EP_gD
            )

        init_feats, edge_i_j_feats = self.Feature_gens(
            y_noise_real, H_real, sigma2, x_real
        )

        if self.compare:
            SER_ML = self.soft_labels(
                x_real, y_noise_real, H_real, sigma2,
                init_feats, edge_i_j_feats
            )

        edge_i_j_feats = edge_i_j_feats.reshape(self.batch_size, -1)

        # Convert constellation values to class indices for CrossEntropyLoss.
        for idx, value in enumerate(self.constellation):
            indices = np.where(x_real == value)
            x_real[indices] = idx

        return (
            H_real, x_real, y_noise_real, init_feats, edge_i_j_feats,
            sigma2, SER_mmse, SER_EP, SER_ML
        )

    def _generate_mimo_channel(self):
        """Build a block MIMO-OTFS channel.

        Every (rx, tx) antenna pair gets its own independent complex tap
        coefficients. Each OTFS block is an MN x MN effective channel.
        """
        block_size = self.M * self.N
        Hc = torch.zeros(
            (self.Nr * block_size, self.Nt * block_size),
            dtype=torch.complex128
        )

        scale = 1.0 / np.sqrt(self.Nt)

        for rx in range(self.Nr):
            for tx in range(self.Nt):
                chan_coef = (
                    torch.sqrt(self.pow_prof)
                    * np.sqrt(0.5)
                    * (torch.randn(self.taps, dtype=torch.float64)
                       + 1j * torch.randn(self.taps, dtype=torch.float64))
                )
                block = find_effective_H_rect_new(
                    self.delay_taps,
                    self.doppler_taps,
                    chan_coef,
                    self.M,
                    self.N
                ).to(torch.complex128)

                r0 = rx * block_size
                r1 = (rx + 1) * block_size
                c0 = tx * block_size
                c1 = (tx + 1) * block_size
                Hc[r0:r1, c0:c1] = scale * block

        return Hc.numpy()

    @staticmethod
    def _complex_to_real(H):
        Hr = np.real(H)
        Hi = np.imag(H)
        return np.block([[Hr, -Hi], [Hi, Hr]])

    def joint_indices(self, indices):
        real_part, complex_part = np.split(indices, 2, axis=1)
        return (
            len(self.constellation) * real_part + complex_part
        ).astype(int)

    def QAM_const(self):
        mod_n = len(self.constellation) ** 2
        sqrt_mod_n = int(np.sqrt(mod_n))
        real_qam_consts = np.empty(mod_n, dtype=np.int64)
        imag_qam_consts = np.empty(mod_n, dtype=np.int64)

        for i in range(sqrt_mod_n):
            for j in range(sqrt_mod_n):
                index = sqrt_mod_n * i + j
                real_qam_consts[index] = i
                imag_qam_consts[index] = j

        return (
            self.constellation[real_qam_consts],
            self.constellation[imag_qam_consts]
        )

    def calc_perf(self, x_soft):
        real_QAM_const, imag_QAM_const = self.QAM_const()
        x_real, x_imag = np.split(x_soft, 2, -1)

        x_real = np.expand_dims(x_real, -1).repeat(
            real_QAM_const.size, -1
        )
        x_imag = np.expand_dims(x_imag, -1).repeat(
            imag_QAM_const.size, -1
        )

        x_dist = (
            np.power(x_real - real_QAM_const, 2)
            + np.power(x_imag - imag_QAM_const, 2)
        )
        estim_indices = np.argmin(x_dist, axis=-1)

        x_indices = self.joint_indices(self.s)
        return np.sum(x_indices != estim_indices) / x_indices.size

    def soft_labels(self, x_real, y_noise_real, H_real, sigma2,
                    init_feats, edge_i_j_feats):
        edge_i_j_feats = edge_i_j_feats.transpose(0, 2, 1)
        init_feats = init_feats.transpose(0, 2, 1)
        sigma2_scalar = np.mean(sigma2)

        yTh = init_feats[:, 0, :]
        hTh = -0.5 * init_feats[:, 1, :]

        temp_a = []
        for i in range(self.num_tx_real):
            for j in range(self.num_tx_real):
                if i != j:
                    temp_a.append(j)

        slicing1 = x_real[:, temp_a]
        step = self.num_tx_real - 1
        slicing2 = []
        for idx in range(0, step * self.num_tx_real, step):
            slicing2.append(slicing1[:, idx:idx + step, None])
        slicing2 = np.concatenate(slicing2, axis=2)

        hth_xj = edge_i_j_feats * slicing2
        combined = []

        for cons_i in self.constellation:
            phi_x_i = (
                yTh * cons_i - hTh * np.square(cons_i)
            ) / sigma2_scalar
            phi_xi_xj = (
                hth_xj * cons_i
            ) / sigma2_scalar
            combine = phi_x_i + np.sum(phi_xi_xj, axis=1)
            combined.append(np.expand_dims(combine, 2))

        aaa = np.concatenate(combined, axis=2)
        normz = aaa - np.max(aaa, axis=2, keepdims=True)
        normalized = np.exp(normz)
        p_x_y = normalized / (
            np.sum(normalized, axis=2, keepdims=True) + np.finfo(float).eps
        )

        x_hats_indices = np.argmax(p_x_y, axis=2)
        return self.calc_perf(self.constellation[x_hats_indices])

    def MMSE(self, x_real, y_noise_real, H_real, sigma2):
        Hty = np.squeeze(
            np.matmul(
                np.transpose(H_real, [0, 2, 1]),
                np.expand_dims(y_noise_real, 2)
            ),
            axis=2
        )

        HtH = np.matmul(
            np.transpose(H_real, [0, 2, 1]), H_real
        )

        eye = np.eye(H_real.shape[2], dtype=H_real.dtype)[None, :, :]
        reg = np.reshape(sigma2, [-1, 1, 1]) * eye
        HtHinv = np.linalg.inv(HtH + reg)

        xhat = np.squeeze(
            np.matmul(HtHinv, np.expand_dims(Hty, 2)), axis=2
        )

        SER = self.calc_perf(xhat)
        var_MMSE = HtHinv * np.mean(sigma2)
        return SER, xhat, var_MMSE

    def Feature_gens(self, y, H, noiseLevel, x_hats):
        # One node per real-valued transmit variable.
        K = self.num_tx_real
        edge_i_j_feats = np.ones(
            (self.batch_size, K, K - 1), dtype=np.float64
        )

        # H^T y and diagonal(H^T H) work for rectangular MIMO channels.
        yTh = np.matmul(
            np.expand_dims(y, 1), H
        )
        hTh = np.matmul(
            H.transpose(0, 2, 1), H
        )
        diag_hTh = np.diagonal(
            hTh, axis1=1, axis2=2
        )[:, None, :]

        noise_arr = np.tile(
            np.expand_dims(noiseLevel, 2), [1, 1, K]
        )
        init_feats = np.concatenate(
            (yTh, -1.0 * diag_hTh, noise_arr), axis=1
        )
        init_feats = init_feats.transpose(0, 2, 1)

        for u_idx in range(K):
            t = 0
            hu = H[:, :, u_idx]
            for j_idx in range(K):
                if j_idx != u_idx:
                    hj = H[:, :, j_idx]
                    edge_i_j_feats[:, u_idx, t] = -np.sum(
                        hj * hu, axis=1
                    )
                    t += 1

        return init_feats, edge_i_j_feats

    def EP(self, x_real, y_noise_real, H_real, sigma2, num_iter):
        user_num = self.num_tx_real
        lamda = np.ones(
            (H_real.shape[0], user_num), dtype=np.float64
        ) * 2.0
        gamma = np.zeros(
            (H_real.shape[0], user_num), dtype=np.float64
        )
        sigma2_scalar = float(np.mean(sigma2))

        constellation_expanded = np.repeat(
            self.constellation[None, :, None],
            H_real.shape[0], axis=0
        )

        def calculate_mean_var(pyx):
            constellation_t = np.repeat(
                constellation_expanded.transpose(0, 2, 1),
                user_num, axis=1
            )
            mean = np.matmul(pyx, constellation_expanded).squeeze(-1)
            var = np.sum(
                pyx * np.square(constellation_t - mean[:, :, None]),
                axis=2
            )
            return mean, var

        def calculate_pyx(mean, var):
            constellation_t = np.repeat(
                constellation_expanded.transpose(0, 2, 1),
                user_num, axis=1
            )
            var = np.clip(var, 1e-13, None)
            log_pyx = -np.square(
                constellation_t - mean[:, :, None]
            ) / (2.0 * var[:, :, None])
            log_pyx -= np.max(log_pyx, axis=2, keepdims=True)
            p_y_x = np.exp(log_pyx)
            return p_y_x / (
                np.sum(p_y_x, axis=2, keepdims=True)
                + np.finfo(float).eps
            )

        def LMMSE(H, y, lamda, gamma):
            HtH = np.matmul(
                np.transpose(H, [0, 2, 1]), H
            )
            Hty = np.squeeze(
                np.matmul(
                    np.transpose(H, [0, 2, 1]),
                    np.expand_dims(y, 2)
                ),
                axis=2
            )
            diag_lamda = np.zeros_like(HtH)
            np.einsum('ijj->ij', diag_lamda)[...] = lamda
            var = np.linalg.inv(
                HtH + diag_lamda * sigma2_scalar
            )
            mean = np.matmul(
                var,
                np.expand_dims(Hty + gamma * sigma2_scalar, 2)
            ).squeeze(-1)
            return mean, var * sigma2_scalar

        mean_b = np.zeros_like(lamda)

        for _ in range(int(num_iter)):
            mean_mmse, var_mmse = LMMSE(
                H_real, y_noise_real, lamda, gamma
            )

            diag_mmse = np.diagonal(
                var_mmse, axis1=1, axis2=2
            )
            var_ab = diag_mmse / (
                1.0 - diag_mmse * lamda
            ) + np.finfo(float).eps
            var_ab = np.clip(var_ab, 1e-13, None)

            mean_ab = (
                np.diagonal(
                    var_mmse, axis1=1, axis2=2
                )
            )
            mean_ab = (
                np.squeeze(mean_mmse) / mean_ab - gamma
            )
            mean_ab = var_ab * mean_ab

            p_y_x_ab = calculate_pyx(mean_ab, var_ab)
            mean_b, var_b = calculate_mean_var(p_y_x_ab)
            var_b = np.clip(var_b, 1e-13, None)

            lamda_new = (
                1.0 / var_b - 1.0 / var_ab
            )
            gamma_new = (
                mean_b / var_b - mean_ab / var_ab
            )

            bad = lamda_new < 0
            lamda_new[bad] = lamda[bad]
            gamma_new[bad] = gamma[bad]

            lamda = 0.9 * lamda + 0.1 * lamda_new
            gamma = 0.9 * gamma + 0.1 * gamma_new

        return self.calc_perf(mean_b)


def find_effective_H_rect_new(delay_taps, doppler_taps, chan_coef, M, N):
    """Single-link OTFS effective channel.

    This is used as one block of the full Nr x Nt MIMO channel.
    """
    H_rect = torch.zeros(
        (M * N, M * N), dtype=torch.complex128
    )

    num_taps = len(delay_taps)
    for ele1 in range(1, M + 1):
        for ele2 in range(1, N + 1):
            for tap_no in range(num_taps):
                delay = int(delay_taps[tap_no])
                doppler = int(doppler_taps[tap_no])

                if ele1 + delay <= M:
                    eff_ele1 = ele1 + delay
                    add_term = torch.exp(
                        1j * 2 * (torch.pi / M)
                        * (ele1 - 1) * (doppler / N)
                    )
                    int_flag = 0
                else:
                    eff_ele1 = ele1 + delay - M
                    add_term = torch.exp(
                        1j * 2 * (torch.pi / M)
                        * (ele1 - 1 - M) * (doppler / N)
                    )
                    int_flag = 1

                add_term1 = torch.tensor(
                    1.0, dtype=torch.complex128
                )
                if int_flag == 1:
                    add_term1 = torch.exp(
                        torch.tensor(
                            -1j * 2 * torch.pi
                            * ((ele2 - 1) / N),
                            dtype=torch.complex128
                        )
                    )

                eff_ele2 = ((ele2 - 1 + doppler) % N) + 1
                new_chan = (
                    add_term * add_term1 * chan_coef[tap_no]
                )

                row_idx = N * (eff_ele1 - 1) + eff_ele2 - 1
                col_idx = N * (ele1 - 1) + ele2 - 1
                H_rect[row_idx, col_idx] = new_chan

    return H_rect
