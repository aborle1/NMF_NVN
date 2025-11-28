#Script to process tests for dirac3-experiment 1 BUT for sklearn timings
#Here, matrix V is constructed out of an exact multiplication of W and H
import numpy as np
import importlib.metadata
import os
from qci_client import JobStatus, QciClient
from pprint import pprint
import dimod
import itertools
from importlib import reload
from nmf import NMF as NMF_potts
from sklearn.decomposition import NMF as NMF_sklearn
import copy
import time

n = k = 4

for i in range(0,100):
    filename = 'mat_'+str(n)+'x'+str(n)+"_"+str(i)+".npz"
    filepath = "./test_matrices/dirac3_exp1/size_" + str(n) + "x" + str(n) + "/" + filename 
    f = np.load(filepath)

    V = copy.deepcopy(f['V'])
    W = copy.deepcopy(f['W'])
    H = copy.deepcopy(f['H'])

    #Sklearn first
    model = NMF_sklearn(n_components=k,init=None,random_state=0)

    start_time = time.process_time() #take timestamp of before you execute the problem
    W_sklearn = model.fit_transform(V)
    H_sklearn = model.components_
    end_time = time.process_time() #take timestamp of after you finish the execution of the problem
    V_rec_sklearn = W_sklearn @ H_sklearn

    cpu_time = end_time - start_time

    print("---",i,"---")
    print("--Sklearn with default init---")
    abs_error_sklearn = np.linalg.norm(V-V_rec_sklearn)
    print("Absolute error:",abs_error_sklearn)
    rel_error_sklearn = abs_error_sklearn/np.linalg.norm(V)
    print("Relative error:",rel_error_sklearn)
    print("CPU time:", cpu_time)


    #save file
    res_filename = 'sklearn_tm_res_' + 'mat_'+str(n)+'x'+str(n)+"_"+str(i)+".npy"
    res_filepath = "./test_matrices/dirac3_exp1/size_" + str(n) + "x" + str(n) + "/results/" + res_filename 
    np.save(res_filepath,cpu_time)

    #close matrix file

    f.close()
