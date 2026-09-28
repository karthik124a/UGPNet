import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from Data_loader import Data_loader, Data_loader_test
from GEPNet import GEPNet
from GNN import GNN
from Train_Eval_funcs import evaluate, train
from parsers import (
    constellation,
    parsersers_,
    snr_test_for_nt,
    snr_train_for_nt,
)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = vars(parsersers_())

Nr = args["Nr"]
M = args["M"]
N = args["N"]
Nt_list = args["Nt_list"]
num_classes = args["num_classes"]
learning_rate = args["learning_rate"]
num_neuron = args["num_neuron"]
num_su = args["su"]
dropout = args["Dropout"]
num_epochs = args["n_epochs"]
iter_GEPNet = args["iter_GEPNet"]
iter_gnn = args["iter_GNN"]
iter_EP_gD = args["iter_EP_genData"]
batch_size = args["batch_size"]
val_size = args["validation_size"]
total_samples = args["samples"]
saved_all_models = args["saved_all_models"]
compare = args["compare"]

QAM_cardinality = len(constellation) ** 2
n_batches = max(1, int(np.ceil(total_samples / batch_size)))
loss_val_final_prev = float("inf")
loss_avg_prev = float("inf")
best_epoch = 0
n_epoch_reducing_learning_rate = 5

print(
    f"Nr={Nr}, M={M}, N={N}, Nt_list={Nt_list}, epochs={num_epochs}, "
    f"nodes per sample={2 * max(Nt_list) * M * N}"
)

GEPNet_model = GEPNet(iter_GEPNet, num_neuron, constellation, device, dtype).to(device)
gnn_model = GNN(iter_gnn, num_neuron, num_su, num_classes, dropout).to(device)
criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(
    list(gnn_model.parameters()) + list(GEPNet_model.parameters()),
    lr=learning_rate,
    weight_decay=1e-4,
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.91, patience=0, threshold=1e-4
)

name = GEPNet_model.__class__.__name__
model_tag = f"{name}_Nr{Nr}_Nt{Nt_list}_M{M}_N{N}_{QAM_cardinality}QAM"
os.makedirs(f"models/{model_tag}", exist_ok=True)
os.makedirs(f"reports/{model_tag}", exist_ok=True)

with open(f"reports/{model_tag}/log.txt", "w") as f:
    f.write(datetime.now().isoformat() + "\n")
    f.write(str(device) + "\n")

with open(f"reports/{model_tag}/reportHPC.txt", "w") as f:
    f.write(f"training samples: {n_batches * batch_size}\n")
    f.write("epoch,loss,trainAcc\n")

for epoch in range(num_epochs):
    t0 = time.time()
    loss_avg = 0.0
    train_acc_avg = 0.0

    for i in range(n_batches):
        Nt = int(np.random.choice(Nt_list))
        snr_db_min, snr_db_max = snr_train_for_nt(Nt)

        loader = Data_loader(
            Nt, Nr, M, N, batch_size,
            snr_db_min, snr_db_max, constellation
        )
        loss, train_acc, train_SER, avg_SNR = train(
            gnn_model, GEPNet_model, device, loader.getTrainData(),
            optimizer, epoch, criterion, 2 * Nt * M * N, dtype,
            constellation, None
        )

        loss_avg += (loss - loss_avg) / float(i + 1)
        train_acc_avg += (train_acc - train_acc_avg) / float(i + 1)

    print(
        f"epoch={epoch}: loss={loss_avg:.8f}, "
        f"train_acc={train_acc_avg:.8f}"
    )

    with open(f"reports/{model_tag}/reportHPC.txt", "a") as f:
        f.write(f"{epoch},{loss_avg},{train_acc_avg}\n")

    with open(f"reports/{model_tag}/log.txt", "a") as f:
        f.write(f"time per epoch: {time.time() - t0}\n")

    if epoch % n_epoch_reducing_learning_rate == 0:
        val_losses = []
        with open(f"reports/{model_tag}/current_acc.txt", "w") as f:
            for Nt in Nt_list:
                for snr_val in snr_test_for_nt(Nt):
                    n_val_batches = max(1, int(np.ceil(val_size / batch_size)))
                    vals = []

                    for _ in range(n_val_batches):
                        loader = Data_loader_test(
                            Nt, Nr, M, N, batch_size, snr_val,
                            constellation, iter_EP_gD, compare
                        )
                        result = evaluate(
                            gnn_model, GEPNet_model, device,
                            loader.getTestData()[0], criterion,
                            2 * Nt * M * N, dtype, constellation
                        )
                        vals.append(result)

                    val_loss = float(np.mean([v[0] for v in vals]))
                    val_ser = float(np.mean([v[2] for v in vals]))
                    val_losses.append(val_loss)
                    print(
                        f"validation: Nt={Nt}, Nr={Nr}, M={M}, N={N}, "
                        f"SNR={snr_val:.1f}, SER={val_ser:.6f}"
                    )
                    f.write(
                        f"Nt={Nt},Nr={Nr},M={M},N={N},"
                        f"SNR={snr_val},SER={val_ser}\n"
                    )

        loss_val_final = float(np.sum(val_losses))
        lr_scheduler.step(loss_val_final)

        if loss_val_final < loss_val_final_prev or (
            np.isclose(loss_val_final, loss_val_final_prev)
            and loss_avg <= loss_avg_prev
        ):
            torch.save(gnn_model.state_dict(), f"models/{model_tag}/model.pkl")
            torch.save(GEPNet_model.state_dict(), f"models/{model_tag}/GEPNet.pkl")
            loss_val_final_prev = loss_val_final
            loss_avg_prev = loss_avg
            best_epoch = epoch

    with open(f"reports/{model_tag}/best_epoch.txt", "w") as f:
        f.write(str(best_epoch))

print(f"Best epoch: {best_epoch}")
