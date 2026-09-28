import argparse
import numpy as np

# Real-axis constellation. For 16-QAM this gives four levels per axis.
num_constell = 16
symblist = [x for x in range(1, 17, 2)]
symPerQadrant = num_constell / 4
list_of_symbols = [symblist[x] for x in range(int(np.sqrt(symPerQadrant)))]
energy_symbols = sum(number ** 2 for number in list_of_symbols)
norm_cons = np.sqrt(2 * np.sqrt(symPerQadrant) * energy_symbols / symPerQadrant)
norm_symbs = list_of_symbols / norm_cons
constellation = np.array(np.concatenate((norm_symbs, -1 * norm_symbs)), dtype=np.float32)

DEFAULT_NR = 4
DEFAULT_NT = 4
DEFAULT_M = 4
DEFAULT_N = 4

# SNR ranges are defined by transmit-antenna count. If an Nt is not explicitly
# listed, the first available training/testing range is used.
snrdb_list_tr = {DEFAULT_NT: np.array([8.0, 30.0])}
snrdb_list_test = {DEFAULT_NT: np.arange(0.0, 35.5, 5.0)}


def parsersers_():
    parser = argparse.ArgumentParser(description="GEPNet detector for true MIMO-OTFS")

    parser.add_argument("--Nr", type=int, default=DEFAULT_NR, help="number of receive antennas")
    parser.add_argument("--Nt_list", type=int, nargs="+", default=[DEFAULT_NT],
                        help="transmit antenna counts used for training")
    parser.add_argument("--Nt_list_test", type=int, nargs="+", default=[DEFAULT_NT],
                        help="transmit antenna counts used for testing")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="OTFS delay dimension")
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="OTFS Doppler/time dimension")

    parser.add_argument("--saved_all_models", action="store_true")
    parser.add_argument("--compare", action="store_true")

    parser.add_argument("--samples", type=int, default=100000, help="training samples per epoch")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--validation_size", "-vs", type=int, default=2000)
    parser.add_argument("--bs_test", type=int, default=128)
    parser.add_argument("--n_epochs", "-ne", type=int, default=40)

    parser.add_argument("--num_neuron", type=int, default=32)
    parser.add_argument("--su", "-su", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--Dropout", type=float, default=0.0)

    parser.add_argument("--num_classes", type=int, default=constellation.shape[0])
    parser.add_argument("--iter_EP_genData", type=int, default=5)
    parser.add_argument("--iter_GEPNet", type=int, default=10)
    parser.add_argument("--iter_GNN", type=int, default=2)

    return parser.parse_args()


def snr_train_for_nt(nt):
    if nt in snrdb_list_tr:
        return snrdb_list_tr[nt]
    return next(iter(snrdb_list_tr.values()))


def snr_test_for_nt(nt):
    if nt in snrdb_list_test:
        return snrdb_list_test[nt]
    return next(iter(snrdb_list_test.values()))
