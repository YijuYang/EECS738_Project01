import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det


fname = sys.argv[1]
feature_num = int(sys.argv[2])
max_K = int(sys.argv[3])
max_epoch = int(sys.argv[4])

fdata = pd.read_csv(fname)
data_col = fdata.get_values()
data_select = np.reshape(data_col[:,feature_num-1], (np.size(data_col,0),1))
data_select = np.sort(data_select.astype(float), axis = 0)

eps_likelihood = 1e-3
likelihood_del = 1
best_run = 0

start_K = 2

def gauss_fun(x, mu, var):
    pd = (1/(np.sqrt(2*np.pi*var)))*np.exp(-1*np.square(x-mu)/(2*var))
    return pd

def em(max_K):
    for K in range (2, max_K+1):
        print('Trying #Gaussians or K =', K)
        epoch = 0
        while epoch < max_epoch:
            for mu_iter in range(0,K):
                idx = np.random.choice(data_select.shape[0], 1, replace=False)
                if mu_iter == 0:
                    idx_all = idx
                else:
                    idx_all = np.append(idx_all, idx)
            mu_hat = np.full((K,1), data_select[idx_all])
            var_hat = np.full((K,1), np.var(data_select))
            pi_hat = np.full((K,1), 1/K)
            gamma_hat = np.empty([np.size(data_select),K])
            iteration = 0
            likelihood_del = 1
            while(likelihood_del > eps_likelihood) and (iteration < 50):
                for k in range(0,K):
                    for n in range(0,np.size(data_select)):
                        # Expectation Step
                        numera_tor = (pi_hat[k,0])*(gauss_fun(data_select[n,0], mu_hat[k,0], var_hat[k,0]))
                        denomina_tor = np.sum((pi_hat[:,0])*(gauss_fun(data_select[n,0], mu_hat[:,0], var_hat[:,0])))
                        gamma_hat[n,k] = numera_tor/denomina_tor
                    # Maximization Step
                    N_k = np.sum(gamma_hat[:,k])
                    mu_hat[k,0] = 0
                    for n in range(0,np.size(data_select)):
                        mu_hat[k,0] = mu_hat[k,0] + (gamma_hat[n,k]*data_select[n,0])
                    mu_hat[k,0] = (1/N_k)*mu_hat[k,0]
                    var_hat[k,0] = 0
                    for n in range(0,np.size(data_select)):
                        var_hat[k,0] = var_hat[k,0] + (gamma_hat[n,k]*((data_select[n,0] - mu_hat[k,0])*(data_select[n,0] - mu_hat[k,0])))
                    var_hat[k,0] = (1/N_k)*var_hat[k,0]
                    pi_hat[k,0] = N_k/np.size(data_select)
                    if iteration == 0:
                        pi_hat_all = pi_hat
                    else:
                        pi_hat_all = np.hstack((pi_hat_all, pi_hat))
                likelihood = 0
                for n in range(0,np.size(data_select)):
                    likelihood = likelihood + np.log(np.sum(pi_hat[:,0]*gauss_fun(data_select[n,0], mu_hat[:,0], var_hat[:,0])))
                if iteration == 0:
                    likelihood_iteration = likelihood
                else:
                    likelihood_iteration = np.append(likelihood_iteration, likelihood)
                if iteration > 0:
                    likelihood_del = np.abs(likelihood_iteration[-1]) - np.abs(likelihood_iteration[-2])
                iteration = iteration + 1
            if epoch == 0:
                likelihood_epoch = likelihood_iteration[-1]
                mu_hat_epoch = mu_hat
                var_hat_epoch = var_hat
                pi_hat_epoch = pi_hat
            else:
                likelihood_epoch = np.append(likelihood_epoch, likelihood_iteration[-1])
                mu_hat_epoch = np.hstack((mu_hat_epoch, mu_hat))
                var_hat_epoch = np.hstack((var_hat_epoch, var_hat))
                pi_hat_epoch = np.hstack((pi_hat_epoch, pi_hat))
            epoch = epoch + 1
        best_likelihood_epoch_idx = np.argmax(likelihood_epoch)
        if K == 2:
            likelihood_K = likelihood_epoch[best_likelihood_epoch_idx]
            mu_hat_K = mu_hat_epoch[:,best_likelihood_epoch_idx]
            var_hat_K = var_hat_epoch[:,best_likelihood_epoch_idx]
            pi_hat_K = pi_hat_epoch[:,best_likelihood_epoch_idx]
        else:
            likelihood_K = np.append(likelihood_K, likelihood_epoch[best_likelihood_epoch_idx])
            mu_hat_K = np.append(mu_hat_K, mu_hat_epoch[:,best_likelihood_epoch_idx])
            var_hat_K = np.append(var_hat_K, var_hat_epoch[:,best_likelihood_epoch_idx])
            pi_hat_K = np.append(pi_hat_K, pi_hat_epoch[:,best_likelihood_epoch_idx])
        K = K + 1
    return likelihood_K, mu_hat_K, var_hat_K, pi_hat_K
