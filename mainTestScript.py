import os
import statistics

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from Data_loader import Data_loader_test
from GEPNet import GEPNet
from GNN import GNN
from Train_Eval_funcs import evaluate
from parsers import constellation, parsersers_, snr_test_for_nt

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = vars(parsersers_())

Nr = args["Nr"]
M = args["M"]
N = args["N"]
Nt_list_train = args["Nt_list"]
Nt_list_test = args["Nt_list_test"]
bs_test = args["bs_test"]
total_samples = args["samples"]
num_classes = args["num_classes"]
num_neuron = args["num_neuron"]
num_su = args["su"]
dropout = args["Dropout"]
iter_GEPNet = args["iter_GEPNet"]
iter_gnn = args["iter_GNN"]
iter_EP_gD = args["iter_EP_genData"]
compare = args["compare"]

QAM_cardinality = len(constellation) ** 2
iter_data = max(1, round(total_samples / bs_test))

GEPNet_model = GEPNet(iter_GEPNet, num_neuron, constellation, device, dtype).to(device)
gnn_model = GNN(iter_gnn, num_neuron, num_su, num_classes, dropout).to(device)
criterion = nn.CrossEntropyLoss().to(device)

model_tag = (
    f"GEPNet_Nr{Nr}_Nt{Nt_list_train}_M{M}_N{N}_{QAM_cardinality}QAM"
)
model_dir = f"models/{model_tag}"

gnn_model.load_state_dict(
    torch.load(f"{model_dir}/model.pkl", map_location=device, weights_only=True)
)
GEPNet_model.load_state_dict(
    torch.load(f"{model_dir}/GEPNet.pkl", map_location=device, weights_only=True)
)
gnn_model.eval()
GEPNet_model.eval()

for Nt in Nt_list_test:
    test_SER = []
    snr_values = []

    for snr in snr_test_for_nt(Nt):
        gep_values = []
        mmse_values = []
        ep_values = []
        ml_values = []

        for _ in range(iter_data):
            loader = Data_loader_test(
                Nt, Nr, M, N, bs_test, snr, constellation,
                iter_EP_gD, compare
            )
            data, ser_mmse, ser_ep, ser_ml = loader.getTestData()
            _, _, gep_ser = evaluate(
                gnn_model, GEPNet_model, device, data, criterion,
                2 * Nt * M * N, dtype, constellation
            )
            gep_values.append(gep_ser)
            if compare:
                mmse_values.append(ser_mmse)
                ep_values.append(ser_ep)
                ml_values.append(ser_ml)

        snr_values.append(float(snr))
        test_SER.append(statistics.mean(gep_values))

        if compare:
            print(
                f"Nt={Nt}, Nr={Nr}, SNR={snr}: "
                f"ML={statistics.mean(ml_values):.8f}; "
                f"MMSE={statistics.mean(mmse_values):.8f}; "
                f"EP={statistics.mean(ep_values):.8f}; "
                f"GEPNet={test_SER[-1]:.8f}"
            )
        else:
            print(
                f"Nt={Nt}, Nr={Nr}, SNR={snr}: "
                f"GEPNet={test_SER[-1]:.8f}"
            )

    report_dir = f"Test_reports/GEPNet_Nr{Nr}_Nt{Nt}_M{M}_N{N}_{QAM_cardinality}QAM"
    os.makedirs(report_dir, exist_ok=True)

    with open(f"{report_dir}/TestReport.txt", "w") as f:
        f.write(f"SNR={snr_values}\n")
        f.write(f"GEPNet={test_SER}\n")
        if compare:
            f.write(f"ML={[statistics.mean(ml_values) for _ in [0]]}\n")
            f.write(f"MMSE={[statistics.mean(mmse_values) for _ in [0]]}\n")
            f.write(f"EP={[statistics.mean(ep_values) for _ in [0]]}\n")

    with open(f"{report_dir}/NumberofTestingData.txt", "w") as f:
        f.write(str(iter_data * bs_test))

    plt.semilogy(snr_values, test_SER, "o-", label="GEPNet")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title(f"{Nt}Tx x {Nr}Rx MIMO-OTFS")
    plt.legend()
    plt.show()
