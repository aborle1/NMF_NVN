#Script to process tests for dirac3-experiment 1 (k = n = m, 1x1, 2x2,3x3,4x4)
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

n = k = 1

for relaxation_schedule in [3]:
    for i in range(88,100):
        filename = 'mat_'+str(n)+'x'+str(n)+"_"+str(i)+".npz"
        filepath = "./test_matrices/dirac3_exp1/size_" + str(n) + "x" + str(n) + "/" + filename 
        f = np.load(filepath)

        V = copy.deepcopy(f['V'])
        W = copy.deepcopy(f['W'])
        H = copy.deepcopy(f['H'])

        #Sklearn first
        model = NMF_sklearn(n_components=k,init=None,random_state=0)
        W_sklearn = model.fit_transform(V)
        H_sklearn = model.components_
        V_rec_sklearn = W_sklearn @ H_sklearn

        print("---",i,"---")
        print("--Sklearn with default init---")
        abs_error_sklearn = np.linalg.norm(V-V_rec_sklearn)
        print("Absolute error:",abs_error_sklearn)
        rel_error_sklearn = abs_error_sklearn/np.linalg.norm(V)
        print("Relative error:",rel_error_sklearn)

        #Now for dirac-3
        url = "https://api.qci-prod.com"
        api_token = "REDACTED"
        size = W.size + H.size  # size here is assuming all Ws and Hs can be 1s
        nmf1 = NMF_potts(V,k,size)
        nmf1.encode_problem()
        nmf1.establish_connection(url,api_token)
        nmf1.run_problem(relaxation_schedule,num_samples=10)
        nmf1.get_result()
        
        #Absolute and relative error values
        print("--Raw Dirac-3 results---")
        abs_error_potts = np.linalg.norm(V-nmf1.V_rec)
        print("Absolute error:",abs_error_potts)
        rel_error_potts = abs_error_potts/np.linalg.norm(V)
        print("Relative error:",rel_error_potts)

        #Fusion
        model = NMF_sklearn(n_components=k,init='custom')
        W_fusion = model.fit_transform(V,W=nmf1.W,H=nmf1.H)
        H_fusion = model.components_
        V_rec_fusion = W_fusion @ H_fusion

        print("--Dirac 3 + sklearn results--")
        abs_error_fusion = np.linalg.norm(V-V_rec_fusion)
        print("Absolute error:",abs_error_fusion)
        rel_error_fusion = abs_error_fusion/np.linalg.norm(V)
        print("Relative error:",rel_error_fusion)

        print("Is fusion better",rel_error_fusion < rel_error_sklearn)

        #Dictionary that we will ultimately save
        result_dict = {}
        #Sklearn results
        result_dict['W_sklearn'] = copy.deepcopy(W_sklearn)
        result_dict['H_sklearn'] = copy.deepcopy(H_sklearn)
        result_dict['V_rec_sklearn'] = copy.deepcopy(V_rec_sklearn)
        result_dict['abs_error_sklearn'] = copy.deepcopy(abs_error_sklearn)
        result_dict['rel_error_sklearn'] = copy.deepcopy(rel_error_sklearn)
        #dirac3 results
        result_dict['W_potts'] = copy.deepcopy(nmf1.W)
        result_dict['H_potts'] = copy.deepcopy(nmf1.H)
        result_dict['V_rec_potts'] = copy.deepcopy(nmf1.V_rec)
        result_dict['abs_error_potts'] = copy.deepcopy(abs_error_potts)
        result_dict['rel_error_potts'] = copy.deepcopy(rel_error_potts)
        #fusion results
        result_dict['W_fusion'] = copy.deepcopy(W_fusion)
        result_dict['H_fusion'] = copy.deepcopy(H_fusion)
        result_dict['V_rec_fusion'] = copy.deepcopy(V_rec_fusion)
        result_dict['abs_error_fusion'] = copy.deepcopy(abs_error_fusion)
        result_dict['rel_error_fusion'] = copy.deepcopy(rel_error_fusion)
        #other dirac3 results
        result_dict['device_usage_s'] = nmf1.job_response['job_info']['job_result']['device_usage_s']
        result_dict['counts'] = nmf1.job_response['results']['counts']
        result_dict['energies'] = nmf1.job_response['results']['energies']
        result_dict['solutions'] = nmf1.job_response['results']['solutions']

        #save file
        res_filename = 'res_' + 'mat_'+str(n)+'x'+str(n)+"_"+str(i)+"_sched_"+str(relaxation_schedule)+".npz"
        res_filepath = "./test_matrices/dirac3_exp1/size_" + str(n) + "x" + str(n) + "/results/" + res_filename 
        np.savez(res_filepath,**result_dict)

        #close matrix file

        f.close()
