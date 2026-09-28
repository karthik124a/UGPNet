import torch
import numpy as np
import torch.nn as nn

soft_max = nn.Softmax(dim=2)


def QAM_const(constellation):
    mod_n = len(constellation) ** 2
    sqrt_mod_n = int(np.sqrt(mod_n))
    real_qam_consts = np.empty(mod_n, dtype=np.int64)
    imag_qam_consts = np.empty(mod_n, dtype=np.int64)

    for i in range(sqrt_mod_n):
        for j in range(sqrt_mod_n):
            index = sqrt_mod_n * i + j
            real_qam_consts[index] = i
            imag_qam_consts[index] = j

    return constellation[real_qam_consts], constellation[imag_qam_consts]


def joint_indices(indices, constellation):
    real_part, imag_part = np.split(indices, 2, axis=1)
    return (len(constellation) * real_part + imag_part).astype(int)


def calc_perf(s, x_soft, constellation):
    real_qam_const, imag_qam_const = QAM_const(constellation)
    x_real, x_imag = np.split(x_soft, 2, -1)

    x_real = np.expand_dims(x_real, -1)
    x_imag = np.expand_dims(x_imag, -1)

    x_dist = (
        np.square(x_real - real_qam_const)
        + np.square(x_imag - imag_qam_const)
    )
    estim_indices = np.argmin(x_dist, axis=-1)

    x_indices = joint_indices(s, constellation)
    return np.mean(x_indices != estim_indices)


def train(
    model, GEPNet, device, train_dataloader, optimizer, epoch, criterion,
    user_num, dtype, constellation, opt=None
):
    model.train()
    GEPNet.train()

    x_hat_list = []
    label_list = []
    sigma2_list = []
    last_loss = None

    for u_feats, edge, label, sigma2, H, y in train_dataloader:
        u_feats = u_feats.to(device=device, dtype=dtype)
        edge = edge.to(device=device, dtype=dtype)
        label = label.to(device=device, dtype=torch.long)
        sigma2 = sigma2.to(device=device, dtype=dtype)
        H = H.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        y_pred = GEPNet(H=H, y=y, sigma2=sigma2, u_feats=u_feats, edge=edge, model=model)
        y_pred_soft = soft_max(y_pred)

        loss = torch.zeros((), dtype=dtype, device=device)
        x_hats = []

        for idx_user in range(user_num):
            loss = loss + criterion(y_pred[:, idx_user, :], label[:, idx_user])

            probs = y_pred_soft[:, idx_user, :].detach().cpu().numpy()
            x_hat = probs @ constellation.reshape(-1, 1)
            x_hats.append(x_hat)

        x_hats = np.concatenate(x_hats, axis=1)
        x_hat_list.append(x_hats)
        label_list.append(label.detach().cpu().numpy())
        sigma2_list.append(sigma2.detach().cpu())

        optimizer.zero_grad()
        if opt is not None:
            opt.zero_grad()
        loss.backward()
        optimizer.step()
        if opt is not None:
            opt.step()

        last_loss = loss

    x_hat_list = np.concatenate(x_hat_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)

    train_SER = calc_perf(label_list, x_hat_list, constellation)
    train_acc = 1.0 - train_SER

    avg_sigma2 = torch.cat(sigma2_list).mean()
    avg_SNR = 10.0 * torch.log10(
        torch.tensor(1.0, dtype=dtype) / (2.0 * avg_sigma2)
    )

    return (
        float((last_loss / user_num).detach().cpu()),
        train_acc,
        train_SER,
        float(avg_SNR),
    )


def evaluate(
    model, GEPNet, device, test_dataloader, criterion,
    user_num, dtype, constellation
):
    model.eval()
    GEPNet.eval()

    x_hat_list = []
    label_list = []
    loss = torch.zeros((), dtype=dtype, device=device)

    with torch.no_grad():
        for u_feats, edge, label, sigma2, H, y in test_dataloader:
            u_feats = u_feats.to(device=device, dtype=dtype)
            edge = edge.to(device=device, dtype=dtype)
            label = label.to(device=device, dtype=torch.long)
            sigma2 = sigma2.to(device=device, dtype=dtype)
            H = H.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            y_pred = GEPNet(H=H, y=y, sigma2=sigma2, u_feats=u_feats, edge=edge, model=model)
            y_pred_soft = soft_max(y_pred)

            x_hats = []
            for idx_user in range(user_num):
                loss = loss + criterion(y_pred[:, idx_user, :], label[:, idx_user])
                probs = y_pred_soft[:, idx_user, :].cpu().numpy()
                x_hats.append(probs @ constellation.reshape(-1, 1))

            x_hat_list.append(np.concatenate(x_hats, axis=1))
            label_list.append(label.cpu().numpy())

    x_hat_list = np.concatenate(x_hat_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)

    test_SER = calc_perf(label_list, x_hat_list, constellation)
    test_acc = 1.0 - test_SER
    return float((loss / user_num).cpu()), test_acc, test_SER
