#Script to process tests for dirac3-experiment 3 (k = n = m, 1x1, 2x2,3x3,4x4)
#Here, matrix V is constructed out of exact factors
import numpy as np
import importlib.metadata
import os
from qci_client import JobStatus, QciClient
from pprint import pprint
import dimod
import itertools
from importlib import reload
from sklearn.decomposition import NMF as NMF_sklearn
from nmf_qubo import *
from nmf import NMF as NMF_potts

n = k = 2
device = 'dirac3'
type = 'qubo' #can take values 'qubo' or 'potts' (which was later renamed to QuarDP in our paper)
for relaxation_schedule in [3]:
    for i in range(0,100):
        print("----",i,"----")
        filename = 'mat_'+str(n)+'x'+str(n)+"_"+str(i)+".npz"
        filepath = "./test_matrices/dirac3_exp3/size_" + str(n) + "x" + str(n) + "/" + filename 
        f = np.load(filepath)
        V = V_real = f['V']
        W_real = f['W']
        H_real = f['H']

        url = "https://api.qci-prod.com"
        api_token = "REDACTED"

        #If we need to run a QUBO problem
        if type == 'qubo':
            power = 2 # max power (2^2 2^1 2^0; 3 bits); values from 0 to 7
            nmf_qubo = NMF_qubo(V,k,power)

            nmf_qubo.encode_problem(2*(np.linalg.norm(V)**2))
            nmf_qubo.establish_connection(url,api_token)
            #IF the Qubo problem needs to be on Dirac-1 (didn't experiment on this for our paper)
            if device == 'dirac1':
                nmf_qubo.run_problem_dirac1(num_samples=10)
            #IF the Qubo problem needs to be on Dirac-3
            elif device == 'dirac3':
                nmf_qubo.run_problem_dirac3(num_samples=10,relaxation_schedule=relaxation_schedule)
        
            nmf_qubo.get_result()

            #Calculate metrics
            abs_error_dirac = np.linalg.norm(V - nmf_qubo.V_rec)
            print("Absolute error:",abs_error_dirac)
            rel_error_dirac = abs_error_dirac/np.linalg.norm(V)
            print("Relative error:",rel_error_dirac)


            #Dictionary that we will ultimately save (here it is only for one machine)
            result_dict = {}
            #dirac3 results
            result_dict['W'] = copy.deepcopy(nmf_qubo.W)
            result_dict['H'] = copy.deepcopy(nmf_qubo.H)
            result_dict['V_rec'] = copy.deepcopy(nmf_qubo.V_rec)
            result_dict['abs_error'] = copy.deepcopy(abs_error_dirac)
            result_dict['rel_error'] = copy.deepcopy(rel_error_dirac)
            #other device related info

            result_dict['device_usage_s'] = nmf_qubo.job_response['job_info']['job_result']['device_usage_s']
            result_dict['counts'] = nmf_qubo.job_response['results']['counts']
            result_dict['energies'] = nmf_qubo.job_response['results']['energies']
            result_dict['solutions'] = nmf_qubo.job_response['results']['solutions']
        

            #Finally save the file
            res_filename = 'res_' + 'mat_'+str(n)+'x'+str(n)+"_"+str(i)+"_"+device +"_sched_"+str(relaxation_schedule)+"_"+str(type)+".npz"
            res_filepath = "./test_matrices/dirac3_exp3/size_" + str(n) + "x" + str(n) + "/results/" + res_filename 
            np.savez(res_filepath,**result_dict)
            f.close()

        else:
            #Use the potts (QuarDP) model with Dirac-3
            #Convert to float for compatibility
            V = V.astype(float)

            m = n
            size =(n*k) + (k*m)  # size here is vestigial (does not apply to integer based running)
            nmf1 = NMF_potts(V,k,size) #We still need to supply size, to not break the code
            nmf1.encode_problem()
            nmf1.establish_connection(url,api_token)

            nmf1.run_problem_integer(max_int = 7, relaxation_schedule=relaxation_schedule,num_samples=10)
            nmf1.get_result()

            #Calculate the metrics
            abs_error_dirac = np.linalg.norm(V - nmf1.V_rec)
            print("Absolute error:",abs_error_dirac)
            rel_error_dirac = abs_error_dirac/np.linalg.norm(V)
            print("Relative error:",rel_error_dirac)

            #Dictionary that we will ultimately save (here it is only for one machine)
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
            res_filename = 'res_' + 'mat_'+str(n)+'x'+str(n)+"_"+str(i)+"_"+device +"_sched_"+str(relaxation_schedule)+"_"+str(type)+".npz"
            res_filepath = "./test_matrices/dirac3_exp3/size_" + str(n) + "x" + str(n) + "/results/" + res_filename 
            np.savez(res_filepath,**result_dict)
            f.close()
