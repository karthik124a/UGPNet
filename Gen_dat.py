import numpy as np
import torch

class genData(object):
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.constellation = params['constellation']
        self.M = params['Nr']
        self.N = params['Nt']
        self.SNR_dB_min = params['SNR_dB_min']
        self.SNR_dB_max = params['SNR_dB_max']
        self.iter_EP_gD = params['iter_EP_gD']
        self.compare = params['compare']
    
    
    def dataTrain(self):
        
        # Generate random indices
        s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size,2 * self.N*self.M])
        Hr = np.zeros((self.batch_size, self.N*self.M, self.N*self.M))
        Hi = np.zeros((self.batch_size, self.N*self.M, self.N*self.M))
        # Map indices to constellations
        self.s=s
        x_real = self.constellation[s]
        for i in range(self.batch_size):
            chan_coef = torch.sqrt(pow_prof)*torch.sqrt(torch.tensor(1/2))*(torch.randn(taps) + 1j*torch.randn(taps))
            Hr[i, :, :] = torch.real(find_effective_H_rect_new(delay_taps, doppler_taps, chan_coef, self.M, self.N))
            Hi[i, :, :] = torch.imag(find_effective_H_rect_new(delay_taps, doppler_taps, chan_coef, self.M, self.N))
        # Generate channel matrix
        H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
        
        # Generate non-corrupted received signal
        y_real = np.squeeze(np.matmul(H_real, np.expand_dims(x_real,2)))
        
        # Generate a range of training SNRs
        snr_db = np.random.uniform(self.SNR_dB_min, self.SNR_dB_max,[self.batch_size,1])
        
        # Calculatet noise variance
        #check once
        sigma2 =  (2 * self.N * self.M) / (np.power(10, snr_db / 10) * (2*self.N*self.M))
        noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.N*self.M)
        
        # Generate corrupted received signal
        y_noise_real = y_real + noise
        
        # Calculate real-valued noise variance
        sigma2 = sigma2/2
        
        SER_mmse            =None
        SER_EP              =None
        SER_ML              =None
        if self.compare:
            # Compute MMSE
            SER_mmse, xhat_MMSE, var_MMSE = self.MMSE(x_real,y_noise_real,H_real,sigma2)
            # Compute EP
            SER_EP =  self.EP(x_real,y_noise_real,H_real,sigma2,self.iter_EP_gD)
            
            
        # Calculate initial and edge features of the GNN
        init_feats,edge_i_j_feats= self.Feature_gens(y_noise_real,H_real,sigma2,x_real)
        if self.compare:
            SER_ML = self.soft_labels(x_real,y_noise_real,H_real,sigma2,init_feats,edge_i_j_feats)
            
        edge_i_j_feats = edge_i_j_feats.reshape(self.batch_size,-1)
        
        
        # Get the symbol indices
        for idx,i in enumerate(self.constellation):
            indices=np.where(x_real==i)
            x_real[indices[0].tolist(),indices[1].tolist()]=idx
            
        return H_real,x_real,y_noise_real,init_feats,edge_i_j_feats,sigma2,SER_mmse,SER_EP, SER_ML


    def joint_indices(self,indices):
        real_part, complex_part = np.split(indices, 2, axis=1)
        joint_indices = (len(self.constellation)*real_part + complex_part).astype(int)
        return joint_indices
            
    def QAM_const(self):
        mod_n = len(self.constellation)**2
        sqrt_mod_n = int(np.sqrt(mod_n))
        real_qam_consts = np.empty((mod_n), dtype=np.int64)
        imag_qam_consts = np.empty((mod_n), dtype=np.int64)
        for i in range(sqrt_mod_n):
            for j in range(sqrt_mod_n):
                    index = sqrt_mod_n*i + j
                    real_qam_consts[index] = i
                    imag_qam_consts[index] = j
                    
        return(self.constellation[real_qam_consts], self.constellation[imag_qam_consts])
    
    
    def calc_perf(self,x_soft):
        
        real_QAM_const,imag_QAM_const = self.QAM_const()
        x_real, x_imag = np.split(x_soft, 2, -1)
        x_real = np.expand_dims(x_real,-1).repeat(real_QAM_const.size,-1)
        x_imag = np.expand_dims(x_imag,-1).repeat(imag_QAM_const.size,-1)
    
        x_real = np.power(x_real - real_QAM_const, 2)
        x_imag = np.power(x_imag - imag_QAM_const, 2)
        x_dist = x_real + x_imag
        estim_indices = np.argmin(x_dist, axis=-1)
        
        
        x_indices = self.joint_indices(self.s)
        ser = np.sum(x_indices!=estim_indices)/x_indices.size
        return ser
    
    def soft_labels(self,x_real,y_noise_real,H_real,sigma2,init_feats,edge_i_j_feats):
        edge_i_j_feats = edge_i_j_feats.transpose(0,2,1)
        init_feats = init_feats.transpose(0,2,1)
        sigma2 = np.mean(sigma2)
        yTh = init_feats[:,0,:]
        hTh = -0.5* init_feats[:,1,:]
        p_x = 1/self.constellation.shape[0]
        
        temp_a = []
        slicing2 = []
        p_x_y=[]
        combined = []
        predicted=[]
        for i in range(self.N*self.M*2):
            for j in range(self.N*self.M*2):
                if  i!=j:
                    temp_a.append(j)
                    
        slicing1 = x_real[:,temp_a]
        
        step = self.N*self.M*2-1
        for idx in range(0,step*self.N*self.M*2,step):
            slicing2.append(slicing1[:,idx:idx+step,None])
        slicing2 = np.concatenate(slicing2,axis=2)

        hth_xj = np.multiply(edge_i_j_feats,slicing2)
        
        for idx,cons_i  in enumerate(self.constellation):
            
            phi_x_i = 1/sigma2 *(yTh * cons_i -  hTh * np.square(cons_i)) 
            phi_xi_xj = (1/sigma2) * hth_xj * cons_i
            sum_phi_xi_xj = np.sum(phi_xi_xj, axis=1)
            combine = phi_x_i +  sum_phi_xi_xj
            combined.append(np.expand_dims(combine,2)) 
            
        aaa = np.concatenate(combined,axis = 2)
        normz =aaa - np.expand_dims(np.max(aaa, axis= 2),2).repeat(len(self.constellation),2)
        normalized = np.exp(normz) 
        p_x_y=normalized/np.expand_dims(normalized.sum(axis=2),2)
        
        x_reals=x_real.copy()
        x_hats_indices = np.argmax(p_x_y,2)
        SER = self.calc_perf(self.constellation[x_hats_indices])
        
        
        # print("Maximum likelihhod SER :",SER)
        return SER

    def MMSE(self,x_real,y_noise_real,H_real,sigma2):
        # Projected channel output
        Hty = np.squeeze(np.matmul(np.transpose(H_real,[0,2,1]), np.expand_dims(y_noise_real,2)))

        # Gramian of transposed channel matrix
        HtH = np.matmul(np.transpose(H_real,[0,2,1]), H_real)
        # Inverse Gramian
        HtHinv = np.linalg.inv(
            HtH + np.reshape(sigma2, [-1, 1,1]) * np.expand_dims(np.eye(H_real.shape[2]),0).repeat(H_real.shape[0], axis=0))

        # MMSE Detector
        xhat = np.squeeze(np.matmul(HtHinv, np.expand_dims(Hty,2)))
        
        SER = self.calc_perf(xhat)
        
        var_MMSE = HtHinv*sigma2[0,0]
        
        return SER,xhat,var_MMSE
    
        

    def Feature_gens(self,y,H,noiseLevel,x_hats):
        # init_feats = np.ones((self.batch_size,self.Nt*2,3))
        edge_i_j_feats = np.ones((self.batch_size,self.N*self.M*2,(self.N*self.M*2-1)))
        
        
        yTh = np.matmul(np.expand_dims(y,2).transpose(0,2,1),H ) 
        hTh = np.matmul(H.transpose(0,2,1),H)
        diag_hTh = np.expand_dims(hTh.diagonal(0,1,2),2).transpose(0,2,1)
        noise_arr = np.tile(np.expand_dims(noiseLevel,2),[1,1,self.N*self.M*2])
        init_feats = np.concatenate((yTh,-1*diag_hTh,noise_arr),1) ## I PUT MINUS HERE!!!!!!!!!!!!!
        init_feats = init_feats.transpose(0,2,1)
        
        for u_idx  in range(self.N*self.M*2):
            t=0;
            for j_idx  in range(self.N*self.M*2):
                if np.not_equal(j_idx, u_idx):
                    edge_i_j_feat = -1*np.matmul(np.expand_dims(H[:,:,j_idx],2).transpose(0,2,1) ,np.expand_dims(H[:,:,u_idx],2))
                    edge_i_j_feats[:,u_idx,t] = np.squeeze(edge_i_j_feat);
                    t=t+1;
        
        return init_feats,edge_i_j_feats
        
    def EP(self,x_real,y_noise_real,H_real,sigma2,num_iter):

        user_num = self.N*self.M *2
        lamda_init = np.ones((H_real.shape[0],self.N*self.M*2))*2
        gamma_init = np.zeros((H_real.shape[0],self.N*self.M*2))
        sigma2 = np.mean(sigma2)
        H = H_real
        y = y_noise_real
        constellation_expanded = np.expand_dims(self.constellation, axis=1)
        constellation_expanded= np.repeat(constellation_expanded[None,...],H.shape[0],axis=0)
        
    
        def calculate_mean_var(pyx, constellation_expanded):
            constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
            mean = np.matmul(pyx, constellation_expanded)
            var = np.square(np.abs(constellation_expanded_transpose - mean))
            var = np.multiply(pyx, var) 
            var = np.sum(var, axis=2)
            
            return np.squeeze(mean), var
        
        def calculate_pyx( mean, var, constellation_expanded):
            constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
            arg_1 = np.square(np.abs(constellation_expanded_transpose - np.expand_dims(mean,2)))
            log_pyx = (-1 * arg_1)/(2*np.expand_dims(var,2))
            log_pyx = log_pyx - np.expand_dims(np.max(log_pyx,2),2)
            p_y_x = np.exp(log_pyx)
            p_y_x = p_y_x/(np.expand_dims(np.sum(p_y_x, axis=2),2) + np.finfo(float).eps)
            
            return p_y_x

        def LMMSE( H, y, sigma2, lamda, gamma):
            HtH = np.matmul(np.transpose(H,[0,2,1]), H)
            Hty = np.squeeze(np.matmul(np.transpose(H,[0,2,1]), np.expand_dims(y,2)))
            diag_lamda = np.zeros((HtH.shape[0],user_num,user_num))
            np.einsum('ijj->ij',diag_lamda)[...] = lamda
            var = np.linalg.inv(HtH + diag_lamda * sigma2)
            mean = (Hty) + gamma* sigma2
            mean = np.matmul(var,np.expand_dims(mean,2))
            var = var* sigma2
            return mean, var

        lamda = lamda_init
        gamma = gamma_init
    
        for iteration in range(num_iter):
                                  
            mean_mmse, var_mmse = LMMSE(H, y, sigma2, lamda, gamma)

            # Calculating mean and variance of P_y_x
            diag_mmse=np.diagonal(var_mmse, axis1=1, axis2=2)
            var_ab = (diag_mmse/ (1 - diag_mmse*lamda )) +np.finfo(float).eps
            
            # var_ab = 1 / (1/np.diagonal(var_mmse, axis1=1, axis2=2) - lamda )
            mean_ab = (np.squeeze(mean_mmse)/np.diagonal(var_mmse, axis1=1, axis2=2) - gamma) 
            mean_ab = var_ab * mean_ab

            # Calculating P_y_x
            p_y_x_ab = calculate_pyx (mean_ab, var_ab, constellation_expanded)

            # Calculating mean and variance of \hat{P}_x_y
            mean_b, var_b = calculate_mean_var(p_y_x_ab, constellation_expanded)
            var_b = np.clip(var_b, 1e-13, None)

            # Calculating new lamda and gamma
            lamda_new = ((var_ab-var_b) / var_b )/ var_ab
            
            # lamda_new = 1/ var_b - 1/var_ab
            gamma_new = mean_b /var_b - mean_ab/ var_ab

            # Avoiding negative lamda and gamma
            if np.any(lamda_new < 0):
                indices = np.where(lamda_new<0)
                lamda_new[indices]=lamda[indices]
                gamma_new[indices]=gamma[indices]

            # Appliying updating weight
            lamda = lamda*0.9 + lamda_new*0.1
            gamma = gamma*0.9 + gamma_new*0.1
            
        
        SER = self.calc_perf(mean_b)
        
        
        return SER

taps = 4
delay_taps = torch.randint(low=0, high=taps, size=(taps,))
doppler_taps = torch.randint(low=0, high=taps, size=(taps,))
pow_prof = (1/taps)*torch.ones(taps)
# chan_coef = torch.sqrt(pow_prof)*torch.sqrt(1/2)*(torch.randn(taps) + 1j*torch.randn(taps))
#Channel matrix for OTFS Modulation
def find_effective_H_rect_new(delay_taps, doppler_taps, chan_coef, M, N):
    # Initialize output matrix with zeros
    H_rect = torch.zeros((M*N, M*N), dtype=torch.complex64)
    taps = len(delay_taps)
    # Iterate through all elements
    for ele1 in range(1, M+1):
        for ele2 in range(1, N+1):
            for tap_no in range(taps):
                # Check if element is within bounds
                if ele1 + delay_taps[tap_no] <= M:
                    eff_ele1 = ele1 + delay_taps[tap_no]
                    add_term = torch.exp(1j * 2 * (torch.pi/M) * (ele1-1) * (doppler_taps[tap_no]/N))
                    int_flag = 0
                else:
                    eff_ele1 = ele1 + delay_taps[tap_no] - M
                    add_term = torch.exp(1j * 2 * (torch.pi/M) * (ele1-1-M) * (doppler_taps[tap_no]/N))
                    int_flag = 1
                
                add_term1 =torch.tensor(1.0, dtype=torch.complex64) 
                # add_term1 = 1
                if int_flag == 1:
                    add_term1 = torch.exp(torch.tensor(-1j * 2 * torch.pi * ((ele2-1)/N)))
                
                eff_ele2 = ((ele2-1+doppler_taps[tap_no]) % N) + 1
                new_chan = add_term * add_term1 * chan_coef[tap_no]
                
                # Convert to 0-based indexing for PyTorch
                row_idx = int(N*(eff_ele1-1)+eff_ele2 - 1)
                col_idx = int(N*(ele1-1)+ele2 - 1)
                H_rect[row_idx, col_idx] = new_chan
                # H_rect[int(N * (eff_ele1 - 1) + eff_ele2 - 1), int(N * (ele1 - 1) + ele2 - 1)] = new_chan
    return H_rect