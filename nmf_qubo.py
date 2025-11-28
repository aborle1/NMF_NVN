#Contains the class for encoding and solving non-negative matrix factorization (NMF) with an Ising/QUBO solver
import copy
import numpy as np
import importlib.metadata
import os
from qci_client import JobStatus, QciClient
from pprint import pprint
import dimod
import itertools
from dwave.samplers import SimulatedAnnealingSampler
import sympy as sp
from qci_client import JobStatus, QciClient



class NMF_qubo:
    def __init__(self,V,k,max_power,**kwargs):
        self.V = copy.deepcopy(V) #Matrix to be factored
        self.max_power = max_power #powers of 2 to describe W and H both (as list)

        self.scale = 1 #/ ((2 ** (max_power + 1)) - 1) #Assuming W and H are going to be in [0,1] 

        self.k = k #number of columns of W and rows of H (number of basis vectors to factor V into)

        self.n = V.shape[0] #number of rows in V and W
        self.m = V.shape[1] #number of columns in V and H

        #Create an empty W
        self.W = np.zeros((self.n,self.k))

        #Create an empty H
        self.H = np.zeros((self.k,self.m))

        #Create V_rec, a reconstruction of V
        self.V_rec = np.zeros((self.n,self.m))

        #Checking if we need to cap wh to a upper value
        if 'wh_constraint' in kwargs and kwargs['wh_constraint'] == True:
            self.wh_constraint = True
            if 'p' in kwargs: #p determines the maximum a w or h element can be
                self.p = kwargs['p'] 
            else:
                self.p = 1.0
            if 'r' in kwargs: #r is the penalty factor for this constraint
                self.r = kwargs['r'] 
            else:
                self.r = 1.0
        else:
            self.wh_constraint = False

        #Checking if we need regularization 
        if 'regularization' in kwargs and kwargs['regularization'] == True:
            self.regularization = True
            if 'alpha_W' in kwargs: #the regularization hyper parameter for W
                self.alpha_W = kwargs['alpha_W']
            else:
                self.alpha_W = 1e-2
            if 'alpha_H' in kwargs: #the regularization hyper parameter for H
                self.alpha_H = kwargs['alpha_H']
            else:
                self.alpha_H = 1e-2
    
    #Encodes the problem as a QUBO function
    def encode_problem(self,quad_penalty): #quad_pentaly is the penalty strength that is need in the process of making Q_init quadratic
        #First add_vars
        self.add_vars()
        #Then prepare the hubo problem (higher order binary) as Q_init
        self.prep_hubo()
        #Making Q_init into quadratic Q
        self.quadratize(quad_penalty)
    
    #This method configures the labels for the variables being used
    def add_vars(self):
        #Adding variables in vars dict
        self.vars = []
        #W
        for i in range(self.n):
            for j in range(self.k):
                for l in range(self.max_power,-1,-1):
                    self.vars.append("wr"+str(i)+str("c")+str(j)+"e"+str(l))
        #H
        for i in range(self.k):
            for j in range(self.m):
                for l in range(self.max_power,-1,-1):
                    self.vars.append("hr"+str(i)+str("c")+str(j)+"e"+str(l))
        
        #Now for var_dict
        #Prepare var_dict
        self.var_dict = {}
        self.var_ctr = 1
        for var in self.vars:
            self.var_dict[var] = {'num': self.var_ctr, 'sym': sp.symbols(var)}
            self.var_ctr += 1
        #Remember, no slack variable needed here
        

        #Do the same for  V
        self.v_vars = []
        self.v_vars_dict = {}
        for i in range(self.n):
            for j in range(self.m):
                self.v_vars.append('vr'+str(i)+"c"+str(j))
                self.v_vars_dict['vr'+str(i)+"c"+str(j)] = self.V[i,j]

    #def get_expressions(self): #Get sympy expressions

    #Create a binary cost  function with degree > 2 (hubo), which NMF problems are first translated into
    #Here, our hubo problem has a degree of 4
    def prep_hubo(self):
        self.Q_init = {} #Q_init is the hubo cost function (before the Qubo's Q dict, we have Q_init)
        self.totexpr = 0.0 #stores entire expression, just incase we need it
        for i in range(self.n):
            for j in range(self.m):
                tempvstr = "vr"+str(i)+"c"+str(j)
                tempexpr = self.v_vars_dict[tempvstr] #Start the expression
                wexpansion_list = []
                hexpansion_list = []
                for l in range(self.k):
                    #getting the first part of the w and h variables
                    tempwstr1 = "wr"+str(i)+"c"+str(l)
                    tempwsymb1 = sp.symbols(tempwstr1) 
                    temphstr1 = "hr"+str(l)+"c"+str(j)
                    temphsymb1 = sp.symbols(temphstr1)
                    #add to tempexpr
                    tempexpr -= tempwsymb1*temphsymb1
                    #print("Expression before subs:",tempexpr) <--- print statement
                    #Now for the things inside
                    wexpansion_expr = 0
                    hexpansion_expr = 0
                    
                    #preparing binary variables
                    for o in range(self.max_power,-1,-1):
                        wexpansion_expr += (2**o)*self.var_dict[tempwstr1+"e"+str(o)]['sym']
                        wexpansion_list.append(self.var_dict[tempwstr1+"e"+str(o)]['sym'])
                        hexpansion_expr += (2**o)*self.var_dict[temphstr1+"e"+str(o)]['sym']
                        hexpansion_list.append(self.var_dict[temphstr1+"e"+str(o)]['sym'])

                    #substitute generic variable with the binary variables
                    tempexpr = sp.expand(tempexpr.subs({tempwsymb1: wexpansion_expr, temphsymb1: hexpansion_expr}))
                    #print("Expression after subs:",tempexpr) <--- print statement

                #once all wh components are ready for vij, then square the expression and expand it
                tempexpr = sp.expand(tempexpr**2)
                #print("Expression after square:",tempexpr) <--- print statement

                #Finally, enforce idempotency w^2 = w and/or h^2 = h
                for item in wexpansion_list:
                    tempexpr = tempexpr.subs(item**2, item)
                for item in hexpansion_list:
                    tempexpr = tempexpr.subs(item**2, item)
                
                #print("Expression after idempotency:",tempexpr) <--- print statement
                #print("---")
                self.tempexpr = tempexpr
                self.totexpr += tempexpr
                #Now for parsing through the expression
                for term in tempexpr.as_ordered_terms():
                    #Get coefficient
                    if term.args != (): #if not constant
                        if isinstance(term.args[0], sp.Symbol): #The 0th element is symbol
                            coeff =  1 #set coefficient as 1
                            start_idx = 0 #which position should variable start at
                        else:
                            coeff = term.args[0]
                            start_idx = 0 #which position should variable start at
                        w_list = []#for w variables
                        h_list = []#for h variables

                        #figure out if var is w or h (within w or h; they would be sorted)
                        for var in term.args[start_idx:]:
                            if 'w' in str(var):
                                w_list.append(str(var))
                            elif 'h' in str(var):
                                h_list.append(str(var))
                        
                        #Make key for Q_init data entry
                        key = tuple(w_list + h_list) 
                        self.Q_init[key] = float(coeff)
                        




    def run_problem_bfhubo(self):
        #Runs Q_init (hubo) as is with brute force
        reversed_dict = {}
        for key,val in self.var_dict.items():
            #Go both ways
            reversed_dict[key] = val['num']
            reversed_dict[val['num']] = key
        print(reversed_dict)

        min_energy = float('inf')
        best_config = None
        for i in range(2**(len(self.var_dict))):
            bitstr = '0'*(len(self.var_dict) -  len(bin(i)[2:]))  + bin(i)[2:] 
            bit_list = [int(x) for x in bitstr]
            bit_energy = 0.0
            for key,val in self.Q_init.items():
                term_energy = val #start with coeff
                for item in key:
                    term_energy *= bit_list[reversed_dict[item] - 1]
                
                bit_energy += term_energy

            if min_energy > bit_energy:
                min_energy = bit_energy
                best_config = list(bit_list)
        
        print(best_config,":",min_energy)
        

    
    def quadratize(self,quad_penalty):
        #Invoke dimod's make_quadratic method (uses Rosenberg's polynomial to break higher-order polynomials to quadratic)
        bqm = dimod.make_quadratic(self.Q_init,quad_penalty,dimod.BINARY)
        self.bqm = bqm #For testing purposes (running bqm directly) CAN REMOVE LATER!!!

        #Put new variables that make_quadratic generated into var_dict
        for key in bqm.linear:
            #Check if key already in self.var_dict
            if key not in self.var_dict:
                #If not in var_dict, put it with ctr
                self.var_dict[key]  = {'num':self.var_ctr, 'sym':None}
                self.var_ctr += 1

        #Create two final Q dicts: one is for string keys, one is for number key
        self.Q_str = {}
        self.Q_num = {}
        #Add a symmetric matrix
        self.Q_mat = np.zeros((self.var_ctr-1,self.var_ctr-1))

        #linear coefficients are added as a doubled entry in key
        for key,value in bqm.linear.items():
            self.Q_str[(key,key)] = value
            self.Q_num[(self.var_dict[key]['num'],self.var_dict[key]['num'])]  = value
        
        #Adding quadratic coefficients
        for key,value in bqm.quadratic.items():
            #for Q_str, the key can be placed as is
            self.Q_str[key] = value

            #for Q_num we need to first decipher key[0] and key[1] into their respective integers 
            self.Q_num[tuple(sorted(( self.var_dict[key[0]]['num'] , self.var_dict[key[1]]['num'] ))) ] = value
        
        #Finally, add it to the symmetric matrix symmetric matrix
        for key,value in self.Q_num.items():
            # for Q_num[i,i]
            if key[0] == key[1]:
                self.Q_mat[key[0]-1,key[1]-1] = value
            else:
                self.Q_mat[key[0]-1,key[1]-1] = value/2
                if (key[1],key[0]) in self.Q_num:
                    print("DANGER! ") #For diagnostic purposes only 
                self.Q_mat[key[1]-1,key[0]-1] = value/2

    #Running the problem with the simmulated annealing solver (CPU)
    def run_problem_sa(self,num_reads=None, num_sweeps = None, num_sweeps_per_beta=1): #Utilizes D-wave's simulated annealing sampler
        #Set solver 'flag` to sim_anneal`
        self.solver = 'sim_anneal'

        sampler = SimulatedAnnealingSampler()
        sampleset =sampler.sample_qubo(self.Q_num,num_reads=num_reads,num_sweeps=num_sweeps, num_sweeps_per_beta=num_sweeps_per_beta)
        self.sample = sampleset.first.sample
        self.sorted_arr = np.sort(sampleset.record, order='energy') #the records in a sorted order (low to high), just in case we need it
    
    #For debugging and diagnostic purposes (on the exact/brute force solver)
    def run_problem_exact(self): #Utilzes D-wave's exact solver sampler
        self.solver = 'exact'
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample_qubo(self.Q_num)
        self.sample = sampleset.first.sample

        self.sorted_arr = np.sort(sampleset.record, order='energy') #the records in a sorted order (low to high), just in case we need it


    #Gets W and H from the results
    def get_result(self):
        #If simulated annealing was used
        if self.solver == 'sim_anneal' or self.solver =='exact':
            #Get W
            for i in range(self.n):
                for j in range(self.k):
                    for o in range(self.max_power,-1,-1):
                        tempwstr = "wr"+str(i)+"c"+str(j)+"e"+ str(o) #Creating variable label to locate correct variable
                        
                        self.W[i,j] += (self.scale)*(2**o) * self.sample[self.var_dict[tempwstr]['num']]

            #Get H
            for i in range(self.k):
                for j in range(self.m):
                    for p in range(self.max_power,-1,-1):
                            temphstr = "hr"+str(i)+"c"+str(j)+"e"+ str(p) #Creating variable label to locate correct variable
                            #print(temphstr)
                            self.H[i,j] += (self.scale)*(2**p)* self.sample[self.var_dict[temphstr]['num']]

        #If QCi devices were used
        elif self.solver == 'dirac-1' or self.solver == 'dirac-3':
            solution = self.job_response['results']['solutions'][0] #extract answer
            #Get W
            for i in range(self.n):
                for j in range(self.k):
                    for o in range(self.max_power,-1,-1):
                        tempwstr = "wr"+str(i)+"c"+str(j)+"e"+ str(o) #Creating variable label to locate correct variable
                        #print(solution[self.var_dict[tempwstr]['num']-1])
                        self.W[i,j] += (self.scale)*(2**o) * solution[self.var_dict[tempwstr]['num']-1]

            #Get H
            for i in range(self.k):
                for j in range(self.m):
                    for p in range(self.max_power,-1,-1):
                            temphstr = "hr"+str(i)+"c"+str(j)+"e"+ str(p) #Creating variable label to locate correct variable
                            #print(solution[self.var_dict[temphstr]['num']-1])
                            self.H[i,j] += (self.scale)*(2**p)* solution[self.var_dict[temphstr]['num']-1]
        
        #Get V_rec
        self.V_rec = self.W @ self.H #reconstruction matrix
    #Establish connection to QCI
    def establish_connection(self,url,api_token):
        self.url = url
        self.api_token = api_token
        self.client = QciClient(url=self.url, api_token=self.api_token) #<--- We will use client in other functions

    
    def run_problem_dirac1(self,num_samples=5,device_type='dirac-1',**kwargs):
        self.solver = device_type
        
        if 'alpha' in kwargs:
            job_params = job_params={"device_type": device_type, "num_samples": num_samples, "alpha": kwargs['alpha']}
        else:
            job_params={"device_type": device_type, "num_samples": num_samples}

        #Create a file for QCI
        qubo_data = {'file_name': "smallest_objective.json",
                     'file_config': {'qubo':{"data": self.Q_mat}}}


        #Create a job and process it
        response_json = self.client.upload_file(file=qubo_data)
        job_body = self.client.build_job_body(job_type="sample-qubo",
                                  qubo_file_id=response_json['file_id'],
                                  job_params=job_params)
        self.job_response = self.client.process_job(job_body=job_body)

        print(
            f"solution: {self.job_response['results']['solutions'][0]} with " 
            f"energy: {self.job_response['results']['energies'][0]}")
    
    def run_problem_dirac3(self,relaxation_schedule=1,num_samples=1):
        self.solver = 'dirac-3'
        #----- PRE- PREPARE PROBLEM
        #First we need to convert the encoding into the appropriate format acceptable for dirac-3
        #Setting variables
        self.poly_indices = [] #List to handle the variables in each term of the polynomial
        self.poly_coefs = [] #List to handle the corresponding coefficients

        #Now, going over the QUBO
        for key,value in self.Q_num.items():
            self.poly_indices.append([key[0],key[1]])
            self.poly_coefs.append(value)
        
        num_levels = []
        for i in range(len(self.var_dict)):
            num_levels.append(2)
        
        #PREPARE FILE OBJECT
        data_int_problem = [{"idx": idx, "val": val} for idx, val in zip(self.poly_indices, self.poly_coefs)]
        file_int_problem = {
        "file_name": "dirac_3_integer_example",
        "file_config": {
        "polynomial": {
            "num_variables": len(self.var_dict),
            "min_degree": 2,
            "max_degree": 2,
            "data": data_int_problem,
        }
        }
        }

        file_response_int_problem = self.client.upload_file(file=file_int_problem)

        self.job_body = self.client.build_job_body(
        job_type='sample-hamiltonian-integer',
        job_name='test_integer_variable_hamiltonian_job', # user-defined string, optional
        job_params={
            'device_type': 'dirac-3',
            'num_samples': num_samples,
            'relaxation_schedule': relaxation_schedule,
            'num_levels': num_levels,  # For demonstration, this excludes some but not all of the known local minima.
        },
        polynomial_file_id=file_response_int_problem['file_id'],
        )


        #----Running the job
        self.job_response = self.client.process_job(job_body=self.job_body)
        # Before inspecting solution, ensure that job did not error.
        assert self.job_response["status"] == JobStatus.COMPLETED.value
        # Only one sample taken.
        print(
            f"solution: {self.job_response['results']['solutions'][0]} with " 
            f"energy: {self.job_response['results']['energies'][0]}"
        )


    def calculate_energy(self,**kwargs):
        #Diagnostic function to check if energy calculation is correct (primarily for QCI)
        #Calculates energy on a solution
        #Currently only implemented on dirac-1
        if self.solver == 'dirac-1' or self.solver == 'dirac-3':
            energy = 0.0
            sol_array = kwargs['sol_array']

            for i in range(self.Q_mat.shape[0]):
                for j in range(self.Q_mat.shape[1]):
                    energy += self.Q_mat[i,j] * sol_array[i]*sol_array[j]
            
            return energy
        elif self.solver == 'dirac-3' and 'Q' in kwargs:
            energy = 0.0
            sol_array = kwargs['sol_array']
            
            for key, value in self.Q_num.items():
                energy += key[0] * key[1] * value
            
            return energy
                



        
        
        elif self.solver == 'dirac-3':
            energy = 0.0
            sol_array = kwargs['sol_array']

            for i in range(len(self.poly_indices)):
                key1, key2 = self.poly_indices[i]
                energy += sol_array[key1-1] * sol_array[key2-1] * self.poly_coefs[i]
            
            return energy
        

    
    def calculate_energy_matrix(self,W,H,which_Q='Q_num'):
        """
        Converts W and H provided into the solution for qubo and then calculates energy from that
        To see if the solution return by solver has an energy higher or lower than the (ideal) W and H
        If solver's answer is wrong BUT it returns better (smaller) or equal energy than the energy calculated using this function
        it means that the formulation is wrong somewhere.
        """

        #Creating a dictionary for substituion
        sub_dict ={}
        #for W
        for i in range(self.n):
            for j in range(self.k):
                #Calculate the bitstring out of W (assume W has integers). Pad the bitstring with appropriate 0s (wrt max_power)
                bitstr = '0'*((self.max_power+1) -  len(bin(W[i,j])[2:]))  + bin(W[i,j])[2:] 
                #print(bitstr)
                for o in range(self.max_power,-1,-1):
                    tempwstr = "wr"+str(i)+"c"+str(j)+"e"+ str(o)
                    #print(tempwstr)
                    sub_dict[tempwstr] = int(bitstr[self.max_power-o])

        #for H
        for i in range(self.k):
            for j in range(self.m):
                #Calculate the bitstring out of H (assume H has integers). Pad the bitstring with appropriate 0s (wrt max_power)
                bitstr = '0'*((self.max_power+1) -  len(bin(H[i,j])[2:]))  + bin(H[i,j])[2:] 
                #print(bitstr)
                for o in range(self.max_power,-1,-1):
                    temphstr = "hr"+str(i)+"c"+str(j)+"e"+ str(o)
                    #print(temphstr)
                    sub_dict[temphstr] = int(bitstr[self.max_power-o])
        print(sub_dict)

        energy = 0.0 #Initialize energy (useful in all the cases)
        #Figure out which Qubo to calculate energy from? Q_str, Q_num or Q_mat?
        if which_Q == 'Q_str':
            print('Q_str')
            for key,value in self.Q_str.items():
                temp_energy = value
                for item in key:
                    if '*' in item:
                        varlist = item.split('*')
                        for item2 in varlist:
                            temp_energy *= sub_dict[item2]
                    else:
                        temp_energy *= sub_dict[item]
                
                energy += temp_energy
            return energy
        
        else:
            #This part is common for both Q_num and Q_mat
            #From var_dict, we are going to prepare a reversed dict but only for the num part of the value
            reversed_dict = {}
            for key,val in self.var_dict.items():    
                reversed_dict[val['num']] = key

            if which_Q == 'Q_num':
                print('Q_num')
                for key,value in self.Q_num.items():
                    temp_energy = value
                    for item in key:
                        if '*' in reversed_dict[item]:
                            varlist = reversed_dict[item].split('*')
                            for item2 in varlist:
                                temp_energy *= sub_dict[item2]
                        else:
                            temp_energy *= sub_dict[reversed_dict[item]]
                    
                    energy += temp_energy
                return energy
            elif which_Q == 'Q_mat':
                for i in range(self.Q_mat.shape[0]):
                    for j in range(self.Q_mat.shape[1]):
                        temp_energy = self.Q_mat[i,j]
                        for item in (i+1,j+1):
                            if '*' in reversed_dict[item]:
                                varlist = reversed_dict[item].split('*')
                                for item2 in varlist:
                                    temp_energy *= sub_dict[item2]
                            else:
                                temp_energy *= sub_dict[reversed_dict[item]]
                        energy += temp_energy
                return energy
            elif which_Q == 'poly':
                for i in range(len(self.poly_indices)):
                    key = self.poly_indices[i]
                    value = self.poly_coefs[i]
                    temp_energy = value
                    for item in key:
                        if '*' in reversed_dict[item]:
                            varlist = reversed_dict[item].split('*')
                            for item2 in varlist:
                                temp_energy *= sub_dict[item2]
                        else:
                                temp_energy *= sub_dict[reversed_dict[item]]
                    energy += temp_energy
                return energy





            


