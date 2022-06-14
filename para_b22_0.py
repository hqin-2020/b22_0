import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
batch_n = np.random.randint(1,10000)
np.set_printoptions(suppress = True)
from Minimization import minimization
from concurrent.futures import ProcessPoolExecutor

n_simulation = 50
start_trials = 14*10

T = 300

λ, η = 0.8, 0
b11, b21 = 1, 0.5

As11, As12, As13, As14 = 0.6, 0.1, 0.1, 9
As21, As22, As23, As24 = 0.2, 0.5, 0.3, 8
As31, As32, As33, As34 = 0.1, 0.2, 0.4, 6

Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = 3, 0.5, 2, 0.3, 0.2, 1

θ_true = np.array([λ, η, \
              b11, b21, \
              As11, As12, As13, As14, \
              As21, As22, As23, As24, \
              As31, As32, As33, As34, \
              Bs11, Bs21, Bs22, Bs31, Bs32, Bs33])

def sim_VAR(A, B, X0, shock_num, T):

    X_series = np.zeros([X0.shape[0], T+1])
    X_series[:, [0]]= X0

    W_series = np.random.multivariate_normal(np.zeros(shock_num), np.diag(np.ones(shock_num)), [T+1]).T

    for t in range(T):
        X_series[:,[t+1]] = A @ X_series[:,[t]] + B @ W_series[:,[t+1]]

    return X_series, W_series

def sim_obs(θ):
    
    λ, η, b11, b21, As11, As12, As13, As14, As21, As22, As23, As24, As31, As32, As33, As34, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33 = θ
    ones = np.ones([3,1])
    Az_true = np.array([[1, 1, 0],\
                        [0, λ, η],\
                        [0, 0, 1]])
    Bz_true = np.array([[b11],\
                        [b21],\
                        [0]])
    As_true = np.array([[As11, As12, As13, As14],\
                        [As21, As22, As23, As24],\
                        [As31, As32, As33, As34],\
                        [0.0,  0.0,  0.0,  1.0]])
    Bs_true = np.array([[Bs11, 0.0,  0.0],\
                        [Bs21, Bs22, 0.0],\
                        [Bs31, Bs32, Bs33],\
                        [0.0,  0.0,  0.0]])
    
    Ass = np.array([[As11, As12, As13],\
                    [As21, As22, As23],\
                    [As31, As32, As33]])
    Aso = np.array([[As14],\
                    [As24],\
                    [As34]])
    Bs =  np.array([[Bs11, 0,    0],\
                    [Bs21, Bs22, 0],\
                    [Bs31, Bs32, Bs33]])
    
    μs = sp.linalg.solve(np.eye(3) - Ass, Aso) 
    Σs = sp.linalg.solve_discrete_lyapunov(Ass, Bs@Bs.T)
        
    μz01 = 0
    Σz01 = 100
    Z01 = np.random.normal(μz01, Σz01)

    μz02 = η/(1-λ)
    Σz02 = b21**2/(1-λ**2)
    Z02 = np.random.normal(μz02, Σz02)

    Z0_true = np.array([[Z01],[Z02],[1]])
    S0_true = np.append(np.random.multivariate_normal(μs.flatten(), Σs),np.array(1)).reshape([-1,1])

    Z_series, Wz = sim_VAR(Az_true, Bz_true, Z0_true, 1, T)
    S_series, Ws = sim_VAR(As_true, Bs_true, S0_true, 3, T)
    obs_series = ones@Z_series[[0],:] + S_series[0:3,:]
    
    return obs_series, Z_series, S_series, Wz, Ws



if __name__ == '__main__':
    batch_results = []
    for i in range(n_simulation):
        obs_series, Z_series, S_series, Wz, Ws = sim_obs(θ_true)
        obs_series = [obs_series for _ in range(start_trials)] 
        with ProcessPoolExecutor() as pool:
            results = pool.map(minimization, obs_series)
        results = [r for r in results]
        batch_results.append(results)

with open('data_'+str(batch_n)+'.pkl', 'wb') as f:
       pickle.dump(batch_results, f)