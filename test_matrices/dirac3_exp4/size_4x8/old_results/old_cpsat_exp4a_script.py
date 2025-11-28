#Use CP-SAT for dirac3 experiment 4 (this assumes you already have dirac-3 results)
import numpy as np
from ortools.sat.python import cp_model
from integer_nmf_ortools import *
import copy
n = 4
m = 8

k= 3
num_samples = 10
relaxation_schedule = 1
device = 'dirac3'
type = 'potts'
thread = 'sthread'
#Just counters for us
dirac3_better = 0
cpsat_better = 0 
for i in range(0,100):
    print("---",i,"---")

    #Get the matrix V
    filename = 'mat_'+str(n)+'x'+str(m)+"_"+str(i)+".npz"
    filepath = "./test_matrices/dirac3_exp4a/size_" + str(n) + "x" + str(m) + "/" + filename 
    f = np.load(filepath)
    V = f['V']
    f.close()
    #Open the dirac-3 results
    res_filename = 'res_' + 'mat_'+str(n)+'x'+str(m)+"_"+str(i)+"_"+device +"_sched_"+str(relaxation_schedule)+"_"+str(type)+".npz"
    res_filepath = "./test_matrices/dirac3_exp4a/size_" + str(n) + "x" + str(m) + "/results/" + res_filename 

    f = np.load(res_filepath)
    #Load W
    W_potts = f['W']
    H_potts = f['H']

    dirac3_time = f['device_usage_s']/10 #Total time div by num_samples
    abs_error_potts = float(f['abs_error'])
    rel_error_potts = float(f['rel_error'])
    f.close()



    print("Time per problem (dirac-3):",dirac3_time)
    rel_error_cpsat_list = []
    W_cpsat_list = []
    H_cpsat_list = []
    #Set the seed value for the first of 10 tries on cp-sat (for each problem it goes from 0 , 1 ... 9)
    seed = 0
    for j in range(num_samples): #num_samples = 10 as dirac3 was run 10 times as welll
        if thread == 'sthread':
            W_cpsat, H_cpsat = integer_nmf_ortools(V, n_components=k,max_time=dirac3_time,init=False,seed=seed,threads=1)
        else:
            W_cpsat, H_cpsat = integer_nmf_ortools(V, n_components=k,max_time=dirac3_time,init=False,seed=seed)
        V_rec_cpsat = W_cpsat @ H_cpsat
        abs_error_cpsat =  np.linalg.norm(V - V_rec_cpsat)
        rel_error_cpsat = abs_error_cpsat/np.linalg.norm(V)
        W_cpsat_list.append(W_cpsat)
        H_cpsat_list.append(H_cpsat)
        rel_error_cpsat_list.append(rel_error_cpsat)
        #Update seed for next try
        seed += 1

    #Get minimum rel_error_cpsat as the representative error value
    rel_error_cpsat = min(rel_error_cpsat_list)
    min_index = rel_error_cpsat_list.index(rel_error_cpsat)
    W_cpsat = copy.deepcopy(W_cpsat_list[min_index])
    H_cpsat = copy.deepcopy(H_cpsat_list[min_index])

    V2 = W_cpsat @ H_cpsat
    rel2 = np.linalg.norm(V - V2)/np.linalg.norm(V)
    

    print("Relative error dirac3:",rel_error_potts)
    print("Relative error cpsat:",rel_error_cpsat)
    print("Double check cpsat:",rel2)

    if rel_error_cpsat < rel_error_potts:
        cpsat_better += 1
    elif rel_error_cpsat > rel_error_potts:
        dirac3_better += 1
    
    #Save current result to a file
    result_dict = {}
    result_dict['W'] = copy.deepcopy(W_cpsat)
    result_dict['H'] = copy.deepcopy(H_cpsat)
    result_dict['V_rec'] = copy.deepcopy(V2)

    result_dict['abs_error'] = copy.deepcopy(np.linalg.norm(V - V2))
    result_dict['rel_error'] = copy.deepcopy(rel_error_cpsat)

    result_dict['W_cpsat_list'] = copy.deepcopy(W_cpsat_list)
    result_dict['H_cpsat_list'] = copy.deepcopy(H_cpsat_list)
    result_dict['rel_error_cpsat_list'] = copy.deepcopy(rel_error_cpsat_list)



    res_filename2 = 'res_' + 'mat_'+str(n)+'x'+str(m)+"_"+str(i)+"_cpsat_"+ str(thread) +".npz"
    res_filepath2 = "./test_matrices/dirac3_exp4a/size_" + str(n) + "x" + str(m) + "/results/" + res_filename2 

    np.savez(res_filepath2,**result_dict)

    print("Times when dirac3 is better:",dirac3_better)
    print("Times when cpsat is better:",cpsat_better)