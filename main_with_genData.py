import torch
import torch.nn as nn
import os
import time
from datetime import datetime
import numpy as np

from GNN import GNN
from parsers import parsersers_,constellation,snrdb_list_tr,snrdb_list_test 
from Train_Eval_funcs import train,evaluate
from Data_loader import Data_loader,Data_loader_test
from GEPNet import GEPNet

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Initialization
args=vars(parsersers_())

saved_all_models = args['saved_all_models']
compare      = args['compare']

total_samples= args['samples']
batch_size   = args['batch_size']
val_size     = args['validation_size']

M           = args['Nr']
Nt_list      = args['Nt_list']
num_classes  = args['num_classes']

learning_rate= args['learning_rate']
beta         = args['beta']
num_neuron   = args['num_neuron']
num_su       = args['su']
dropout      = args['Dropout']

num_epochs   = args['n_epochs']
iter_GEPNet  = args['iter_GEPNet']
iter_gnn     = args['iter_GNN']
iter_EP_gD   = args['iter_EP_genData']
n_epoch_reducingLearningRate= 5

QAM_cardinality = len(constellation)**2
n_batches = int(np.round(total_samples/batch_size))
loss_val_final_prev=float('inf')
loss_avg_prev=None

print(f'M={M},N_list={Nt_list},num_epochs={num_epochs},num_neuron={num_neuron},iter_GEPNet={iter_GEPNet},beta={beta},learning_rate={learning_rate}')
dt_string = datetime.now().strftime("%d_%H:%M:%S")


# Initialize models, optimizer, and learning scheduler
GEPNet = GEPNet(iter_GEPNet, num_neuron,constellation, device, dtype).to(device)
model = GNN(iter_gnn, num_neuron, num_su, num_classes, dropout).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(GEPNet.parameters()), lr=learning_rate, weight_decay=1e-4)
opt = torch.optim.Adam(GEPNet.parameters(), lr=learning_rate, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', 0.91, 0, 0.0001, 'rel', 0, 0, 1e-08)
name=GEPNet.__class__.__name__ 

# Create directories to save trained models and training reports
for k in ['models','reports']:
    if not os.path.exists(k):
        os.makedirs(k)
    if not os.path.exists(f'{k}/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM'):
        os.makedirs(f'{k}/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM')

f = open(f'reports/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/log.txt',"w")
f.write(dt_string + "\n")
f.write(str(device) + "\n")
f.close()
    
f = open(f'reports/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/reportHPC.txt',"w")
f.write("number Of train samples: "+ str(n_batches*batch_size) + "\n")
f.write("number Of batches in an epoch: "+ str(n_batches) + "\n")
f.write("epoch, loss, trainAcc "+ "\n"+ "\n")
f.close()
    
for epoch in range(num_epochs):

    t = time.time()
    loss_avg=0
    train_acc_avg=0
    
    for i in range (n_batches):
        # Randomly pick NT from NT list
        N = np.random.choice(Nt_list)
        
        # Generate training data
        snr_db_min,snr_db_max = snrdb_list_tr[N]
        dataLoader = Data_loader (N, M, batch_size, snr_db_min, snr_db_max, constellation)
        train_dataloader = dataLoader.getTrainData()
        
        # Train GEPNet
        loss,train_acc, train_SER,avg_SNR=train(model,GEPNet,device,train_dataloader,optimizer,epoch,criterion,N*M*2,dtype,constellation, opt)
        loss_avg =  loss_avg + (loss-loss_avg)/float(i+1.0)
        train_acc_avg = train_acc_avg + (train_acc-train_acc_avg)/float(i+1.0)
        
        
    print(f'epoch{epoch}: loss={loss_avg:.8f},train_acc={train_acc_avg:.8f} ')
    
    f = open(f'reports/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/reportHPC.txt',"a")
    f.write(str(epoch) + "," + str(loss_avg) + "," + str(train_acc_avg)  + "\n")
    f.close()    
        
    elapsed = time.time() - t
    f = open(f'reports/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/log.txt',"a")
    f.write("time per epoch: " + str(elapsed) + "\n")
    f.close()
    
    
    
    # Validation 
    if epoch%n_epoch_reducingLearningRate==0:
        
        print('\n******************************** Now Validating **********************************************')

        loss_val_list=[]
        loss_val_lists=[]
        Nt = np.random.choice(Nt_list)
        n_batches_val = int(np.round(val_size/batch_size))
        snr_val = snr_db_max
        f = open(f'reports/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/current_acc.txt', "w")
        for N in Nt_list:
            for snr_val in snrdb_list_test[N]:
                SER_mmse_avg=0
                SER_EP_avg=0
                SER_ML_avg=0
                val_acc_avg=0
                val_SER_avg=0
                loss_val_avg=0
                for val_idx in range(n_batches_val) :
                    dataLoader_val= Data_loader_test (N,M,batch_size,snr_val,constellation,iter_EP_gD,compare)
                    valid_dataloader,SER_mmse,SER_EP, SER_ML= dataLoader_val.getTestData()
                    loss_val,val_acc,val_SER=evaluate(model,GEPNet,device,valid_dataloader, criterion, N*M*2,dtype,constellation)
                    loss_val_avg =  loss_val_avg + (loss_val-loss_val_avg)/float(val_idx+1.0)      
                    val_acc_avg =  val_acc_avg + (val_acc-val_acc_avg)/float(val_idx+1.0)          
                    val_SER_avg =  val_SER_avg + (val_SER-val_SER_avg)/float(val_idx+1.0)     
                loss_val_list.append(loss_val_avg)
            loss_val_lists.append(np.mean(loss_val_list))
            print(f'val_acc={val_acc_avg:.8f}, val_SER={val_SER_avg:.8f}, snr={snr_val:.2f}, N={N}, M={M} \n')    
            f.write("current SER: "+ str(val_SER_avg) + ",M:" + str(M) + ",NT_val:" + str(N) + ",SNR:"+ str(snr_val)+ ",QAM:" + str(QAM_cardinality)+'\n')
        f.close()
        
        loss_val_final = np.sum(loss_val_lists)
        # optimizer.step(loss_val_final)
        lr_scheduler.step(loss_val_final)

    model.to('cpu')
    GEPNet.to('cpu')
    
# Save trained model
    if saved_all_models: 
        torch.save(model.state_dict(), f'models/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/model_{epoch}.pkl')
        torch.save(GEPNet.state_dict(), f'models/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/GEPNet_{epoch}.pkl')
        best_epoch = 0
    else:
        if (loss_val_final < loss_val_final_prev):
            torch.save(model.state_dict(), f'models/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/model.pkl')
            torch.save(GEPNet.state_dict(), f'models/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/GEPNet.pkl')
            loss_val_final_prev = loss_val_final
            loss_avg_prev = loss_avg
            best_epoch=epoch
        elif (loss_val_final == loss_val_final):
            if loss_avg <= loss_avg_prev:
                torch.save(model.state_dict(), f'models/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/model.pkl')
                torch.save(GEPNet.state_dict(), f'models/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/GEPNet.pkl')
                loss_avg_prev = loss_avg
                best_epoch=epoch
    
    model.to(device)
    GEPNet.to(device)
    f = open(f'reports/{name}_{M}X{Nt_list}_{QAM_cardinality}QAM/best_epoch.txt', "w")
    f.write(str(best_epoch))
    f.close()