
'Implementation of Koopman operator with Fourier feature and testing it with acasxu data samples'


import os
import sys
import numpy as np
from scipy import stats
import acasxu_dubins_koopman as acas_koopman
import matplotlib.pyplot as plt


def generate_simulation_states(num_sims):

    """Generates states through actual simulation of the system.
      Output is a list of matrices where each matrix contains 
      all the states captured during a full simulation.
      Each matrix contains all x, y, theta, cmd generated during
      steps of the simulation"""

    s = acas_koopman.main(num_sims)
    return s


def generate_Fourier(n, count):
    
    """Generating a set of Wi and bi for Fourier features
     to be used in obsevables g()"""

    lw = []
    lb = []
    l = 1
    np.random.seed(0)
 
    for i in range(count):
        WT = stats.norm.rvs(loc=0, scale=l, size=n)  
        b = stats.uniform.rvs(loc=0, scale=2*np.pi, size=1)
        lw.append(WT)
        lb.append(b)
  
    return lw, lb
       


def g(X, WT, b):

    """creating observables g1(x), g2(x), ..., gn(x). 
     We generate them using Fourier feature
     g(X) = cos(wT*X + b)"""
    
    out = np.cos(np.dot(WT, X) + b)
    return out
     


def DMD(X, U, rank):

    'Dynamic Mode Decomposition'

    tmp = X[0]
    X1 = tmp[:,0:tmp.shape[1] -1]
    X2 = tmp[:,1:tmp.shape[1]]
   
    for i in  range(1, len(X)):
       tmp = np.array(X[i])
       X1 = np.concatenate((X1,tmp[:,0:tmp.shape[1] -1]), axis=1)
       X2 = np.concatenate((X2,tmp[:,1:tmp.shape[1]]), axis=1)

    
    U_ = np.array([U[0]])
    
    for i in range(1, len(U)):
        U_ = np.concatenate((U_, np.array([U[i]])), axis=1) 


    X1 = np.concatenate((X1, U_), axis=0)


    #singular value decomposition

    V,S,W = np.linalg.svd(X1, full_matrices=False)
        
    #reduce rank by removing the smallest singular values
    V = V[:, 0:rank]
    S = S[0:rank] 
    W = W[0:rank, :]
    
   
    AA = np.linalg.multi_dot((X2, np.transpose(W), np.diag(np.divide(1, S)), np.transpose(V)))
    
    
    #devide into state matrix and input matrix

    B = AA[:, AA.shape[0] : AA.shape[1]]
    A = AA[:, 0: AA.shape[0]]
    

    return A, B


def generate_xp(s, WF,BF):
    
    out = np.zeros((len(WF), s.shape[1]))
    for i in range(len(WF)): # iterating thorough each wi
        for r in range(s.shape[1]):
            out[i, r] = g(s[:,r], WF[i], BF[i])
 
    res = np.concatenate((s, out))

    return res
    


def predict(xs,us, A, B, num_observables, WF,BF):

    g_xs = generate_xp(xs, WF, BF)
    out = np.dot(A,g_xs) + np.multiply(B, us)
    return out
    


def preprocessing(sim_states, sim_cmds):
  
    'Normalizing and scalling the dataset'

    'concating all arrays in sim_states column-wise to get the norm' 
    tmp_s = np.zeros((sim_states[0].shape[0],1))
    for a in sim_states:
        for j in range(a.shape[1]):
            x = np.transpose(np.array([a[:,j]]))
            tmp_s = np.concatenate((tmp_s, x), axis=1)


    norm_s_x = np.linalg.norm(tmp_s[0, :])
    norm_s_y = np.linalg.norm(tmp_s[1, :])
    norm_s_t = np.linalg.norm(tmp_s[2, :])

    normalized_sim_states = []

    for a in sim_states:   
        a[0,:] = a[0,:] / norm_s_x
        a[1,:] = a[1,:] / norm_s_y
        a[2,:] = a[2,:] / norm_s_t

    
    tmp_c = np.concatenate(sim_cmds)
    norm_c = np.linalg.norm(tmp_c)
    sim_cmds = sim_cmds / norm_c
    

    return sim_states, sim_cmds

    

def Test(init_test_state, test_cmds, A, B, num_observables, WF,BF):
    
    s = init_test_state[:,0]
    cmds = test_cmds
    outx = [s[0]]

    for i in range(len(cmds)):
        tmp = np.transpose(np.array([s]))
        p = predict(tmp, cmds[i], A, B, num_observables, WF, BF)   
        s = p[0:3, 0]
        outx.append(s[0])
    
    'ploting each state and predicted state x, y, tetha'
    plt.plot(outx, '--', color="r")
    plt.plot(init_test_state[0,:], '.', color="b")
    plt.xlabel("number of samples")
    plt.ylabel("Acasxu intruder state: (x, y, theta)")
    plt.legend(["Predicted states", "Simulated states"])
    plt.show()




def main():
    
    sim_nums = 500
    
    sim_states, sim_cmds = generate_simulation_states(sim_nums)

    if len(sim_states) == 0:
        print("Empty training set.")
        return 

    n_sim_states, n_sim_cmds = preprocessing(sim_states, sim_cmds)


    rate = 0.9 # rate of train data
    train_size = int(np.round(sim_nums*rate)) # train dataset size
    test_states = n_sim_states[train_size:] # test
    test_cmds = n_sim_cmds[train_size:] # test
    states = n_sim_states[0: train_size] # train
    cmds = n_sim_cmds[0: train_size] # train
    

 
    'Generate Fourier matrices for count number of observables'
    num_observables = 50
    rank = num_observables
    X = [0, 0, 0]
    XP = []
    WF , BF = generate_Fourier(len(X), num_observables)



    'making input from states and observables for DMD'     
    for s in states:
        xps = generate_xp(s, WF, BF)
        XP.append(xps)
   

    A, B = DMD(XP, cmds, rank) # Koopman operators
    

    for i in range(len(test_states)):
        Test(test_states[i], test_cmds[i], A, B, num_observables, WF, BF)


if __name__ == "__main__":
    main()
