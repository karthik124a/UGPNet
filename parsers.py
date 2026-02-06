import argparse
import numpy as np

num_constell = 16
symblist = [x for x in range(1,17,2)] #max 26244-QAM
symPerQadrant = num_constell/4
list_of_symbols = [symblist[x] for x in range (int(np.sqrt(symPerQadrant)))]
energy_symbols = sum([number ** 2 for number in list_of_symbols])
norm_cons = np.sqrt(2*np.sqrt(symPerQadrant)  * ( energy_symbols) / symPerQadrant)
norm_symbs = list_of_symbols/norm_cons
constellation= np.array(np.concatenate((norm_symbs, -1*norm_symbs)), dtype=np.float32)

# Training configuration
N_tr = 32
snrdb_list_tr = {N_tr:np.array([8.0, 50.5])}    # Only need to specify SNRdB min and max for training
# Testing configuration
N = N_tr
snrdb_list_test = {N:np.arange(0.0, 35.5,5.0)}

def parsersers_():
    parser = argparse.ArgumentParser(description='GNN for detection')
    parser.add_argument('--Nr', type=int, default=64, help='Nr')
    parser.add_argument('--Nt_list', type=int, default=np.array([N_tr]), help='Nt_list')
    parser.add_argument('--Nt_list_test', type=int, default=np.array([N]), help='Nt_list_test')
    
    parser.add_argument('--saved_all_models',default=False, help='saved_all_models')
    parser.add_argument('--compare', default=True, help='compare with other detectors')
    
    parser.add_argument('--samples', type=int, default=10000000, help='train sample data')
    parser.add_argument('--batch_size', type=int, default=128, help='num_samplesPer_batch')#changed 128-64
    parser.add_argument('--validation_size','-vs', type=int, default=20000, help='validation_size')
    parser.add_argument('--bs_test', type=int, default=128, help='TestingBatchSize')#changed 256-64
    parser.add_argument('--n_epochs','-ne', type=int, default=40, help='n_epochs')
    
    parser.add_argument('--num_neuron', type=int, default=32, help='num_neuron')
    parser.add_argument('--su','-su', type=int, default=32, help='num_feature_su')
    parser.add_argument('--beta', type=float, default=0.7, help='beta_EP')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning_rate')
    parser.add_argument('--Dropout', type=float, default=0, help='dropout')
        
    parser.add_argument('--num_classes', type=int, default=constellation.shape[0], help='num_classes')
    parser.add_argument('--iter_EP_genData', type=float, default=5, help='number of iterations for conventional detector e.g., EP')
    parser.add_argument('--iter_GEPNet', type=int, default=10, help='number of GEPNet iterations ')
    parser.add_argument('--iter_GNN', type=int, default=2, help='number of GNN iterations inside a GEPNet iteration')
    return parser.parse_args()

# SNR=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
# ML=[0.37033941806891024, 0.24327987279647437, 0.12578876201923078, 0.04837114383012821, 0.012541942107371794, 0.0027544070512820515, 0.0004851512419871795]
# MMSE=[0.4120592948717949, 0.30693797576121795, 0.20075871394230768, 0.1100604717548077, 0.046640249399038464, 0.016066331129807692, 0.004181690705128205]
# EP=[0.40711701221955127, 0.295131585536859, 0.1763133513621795, 0.07671023637820513, 0.02257987780448718, 0.004904722556089744, 0.0007918920272435897]
# GEPNet=[0.41125175280448717, 0.2955478766025641, 0.1731270032051282, 0.0738681891025641, 0.021775465745192308, 0.004870292467948718, 0.0008012820512820513]