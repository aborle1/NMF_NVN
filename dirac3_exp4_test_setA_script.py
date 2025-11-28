#Script to process tests for dirac3-experiment 4 (k =3 n =4 and m=5,6,7,8) : note: k is p in the paper
#For test set A
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
n = 4
m = 8
k= 3
relaxation_schedule = 1
device = 'dirac3'
type = 'potts' #potts is an alias for QuarDP (from the potts model)
objective = 'l1' #l1 : sum of abs diff, l2 : sum of squared diff
for i in range(0,100):
    print("---",i,"---")
    filename = 'mat_'+str(n)+'x'+str(m)+"_"+str(i)+".npy"
    filepath = "./test_matrices/dirac3_exp4/size_" + str(n) + "x" + str(m) + "/" + filename 
    V = np.load(filepath)

    #Now for dirac-3
    url = "https://api.qci-prod.com"
    api_token = "REDACTED"
    V = V.astype(float)
    size = (n*k) + (k*m) ##Constraint R is the sum of all variables involved in the problem
    nmf1 = NMF_potts(V,k,size)
    nmf1.encode_problem()
    nmf1.establish_connection(url,api_token)
    nmf1.run_problem_integer(max_int = 7, relaxation_schedule=relaxation_schedule,num_samples=10 )
    nmf1.get_result()
    
    #Absolute and relative error values
    abs_error_dirac = np.linalg.norm(V - nmf1.V_rec)
    print("Absolute error:",abs_error_dirac)
    rel_error_dirac = abs_error_dirac/np.linalg.norm(V)
    print("Relative error:",rel_error_dirac)

    #Dictionary that we will ultimately save 
    result_dict = {}
    #dirac3 results
    result_dict['W'] = copy.deepcopy(nmf1.W)
    result_dict['H'] = copy.deepcopy(nmf1.H)
    result_dict['V_rec'] = copy.deepcopy(nmf1.V_rec)
    result_dict['abs_error'] = copy.deepcopy(abs_error_dirac)
    result_dict['rel_error'] = copy.deepcopy(rel_error_dirac)
    #other device related info

    result_dict['device_usage_s'] = nmf1.job_response['job_info']['job_result']['device_usage_s']
    result_dict['counts'] = nmf1.job_response['results']['counts']
    result_dict['energies'] = nmf1.job_response['results']['energies']
    result_dict['solutions'] = nmf1.job_response['results']['solutions']


    #Finally save the file
    res_filename = 'res_' + 'mat_'+str(n)+'x'+str(m)+"_"+str(i)+"_"+device +"_sched_"+str(relaxation_schedule)+"_"+str(type)+".npz"
    res_filepath = "./test_matrices/dirac3_exp4/size_" + str(n) + "x" + str(m) + "/results/" + res_filename 
    np.savez(res_filepath,**result_dict)

