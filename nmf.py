#Contains the class for encoding and solving non-negative matrix factorization (NMF) with Dirac-3
#This is for real-world and integer NMFs
import copy
import numpy as np
import importlib.metadata
import os
from qci_client import JobStatus, QciClient
from pprint import pprint
import dimod
import itertools

class NMF:
    def __init__(self,V,k,sum_constraint,**kwargs):
        self.V = copy.deepcopy(V) #Matrix to be factored
        self.sum_constraint = sum_constraint #sum constraint for Dirac 3
        self.k = k #number of columns of W and rows of H (number of basis vectors to factor V into), this is called p in the paper

        self.n = V.shape[0] #number of rows in V and W
        self.m = V.shape[1] #number of columns in V and H

        #Create an empty W
        self.W = np.zeros((self.n,self.k))

        #Create an empty H
        self.H = np.zeros((self.k,self.m))

        #Create V_rec, a reconstruction of V
        self.V_rec = np.zeros((self.n,self.m))

        #Checking if we need to cap wh to a upper value
        #Note these constraints were not used or tested in the current paper (Future work)
        if 'wh_constraint' in kwargs and kwargs['wh_constraint'] == True:
            self.wh_constraint = True
            if 'p' in kwargs: #p determines the maximum a w or h element can be (NOT the p in the paper)
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
        #Note: regularization was not tested or fully implemented for the current paper (Future work)
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
    
    #Encodes the problem as a QuarDP function
    def encode_problem(self):
        #Adding variables in vars dict
        self.vars = []
        #creating labels for the variables
        #W
        for i in range(self.n):
            for j in range(self.k):
                self.vars.append("w_"+str(i)+str(",")+str(j))
        #H
        for i in range(self.k):
            for j in range(self.m):
                self.vars.append("h_"+str(i)+str(",")+str(j))
        
        #Prepare var_dict
        self.var_dict = {}
        self.var_ctr = 1
        for var in self.vars:
            self.var_dict[var] = self.var_ctr
            self.var_ctr += 1
        #Finally, add the slack variable
        self.var_dict['s0'] = self.var_ctr

        #Do the same for  V
        self.v_vars = []
        self.v_vars_dict = {}
        for i in range(self.n):
            for j in range(self.m):
                self.v_vars.append('v_'+str(i)+","+str(j))
                self.v_vars_dict['v_'+str(i)+","+str(j)] = self.V[i,j]

        #Now for the actual code to formulate the problem
        self.poly_indices = [] #List to handle the variables in each term of the polynomial
        self.poly_coefs = [] #List to handle the corresponding coefficients

        for i in range(self.n):
            for j in range(self.m):
                self.whset = set() #whset need not be an instance variable, consider refactoring
                tempvstr = "v_"+str(i)+","+str(j)
                for l in range(self.k):
                    tempwstr = "w_"+str(i)+","+str(l)
                    temphstr = "h_"+str(l)+","+str(j)
                    # w^2 h^2 type of term
                    self.poly_indices.append([self.var_dict[tempwstr],self.var_dict[tempwstr],self.var_dict[temphstr],self.var_dict[temphstr]]) 
                    self.poly_coefs.append(1)
                    # -2vwh type of term
                    self.poly_indices.append([0,0,self.var_dict[tempwstr],self.var_dict[temphstr]]) 
                    self.poly_coefs.append(-2*self.v_vars_dict[tempvstr])
                    #store tempwstr and temphstr in whset
                    self.whset.add( (tempwstr,temphstr) )
                    
                #Now for preparing whwh type of terms +2whwh
                for combo in itertools.combinations(self.whset,2):
                    #which w is first
                    if combo[0][0] < combo[1][0]:
                        w1 = combo[0][0]
                        w2 = combo[1][0]
                    else:
                        w2 = combo[0][0]
                        w1 = combo[1][0]
                    
                    #which h is first
                    if combo[0][1] < combo[1][1]:
                        h1 = combo[0][1]
                        h2 = combo[1][1]
                    else:
                        h2 = combo[0][1]
                        h1 = combo[1][1]
        
                    self.poly_indices.append([self.var_dict[w1],self.var_dict[w2],self.var_dict[h1],self.var_dict[h2]])
                    self.poly_coefs.append(2)
            
        # +s (slack variable)
        self.poly_indices.append([0,0,0,self.var_dict['s0']])
        self.poly_coefs.append(0)


        #if we have the wh constraints, add them (using encode_constraints function)    
        if self.wh_constraint == True:
            self.encode_constraints()
        #if we need regularization, add them (using encode_regularization)
    
    #Not used in the paper (can be used or extended for Future work)
    def encode_constraints(self):
        #If you wanted to add constraints to ws and hs : say to keep them in [0,p]
        #For example, for W, You need (p - w_i,j - s_w_i,j)^2 where s_w_i,j is a slack variable solely for absorbing additional value from w_i,j s.t they both sum to p
        
        #First add additional slack variables to var_dict
        #Slacks for W
        for i in range(self.n):
            for j in range(self.k):
                self.var_ctr += 1 #var_ctr increments here (because of what happened before in encode_problem)
                tempsstr= 's_w_'+ str(i) + "," + str(j)
                self.var_dict[tempsstr] = self.var_ctr

        #Slacks for H
        for i in range(self.k):
            for j in range(self.m):
                self.var_ctr += 1
                tempsstr = 's_h_'+ str(i) + "," + str(j)
                self.var_dict[tempsstr] = self.var_ctr
        
        #penalties for W
        for i in range(self.n):
            for j in range(self.k):
                #+w^2 * r
                tempwstr = "w_"+str(i)+str(",")+str(j)
                self.poly_indices.append([0, 0, self.var_dict[tempwstr],self.var_dict[tempwstr]])
                self.poly_coefs.append(1*self.r)
                #+s_w^2 * r
                tempsstr= 's_w_'+ str(i) + "," + str(j)
                self.poly_indices.append([0, 0, self.var_dict[tempsstr],self.var_dict[tempsstr]])
                self.poly_coefs.append(1*self.r)
                #-2w * r * p
                self.poly_indices.append([0, 0, 0,self.var_dict[tempwstr]])
                self.poly_coefs.append(-2*self.r*self.p)
                #-2s_w * r * p
                self.poly_indices.append([0, 0, 0,self.var_dict[tempsstr]])
                self.poly_coefs.append(-2*self.r*self.p)
                #+2ws_w * r
                self.poly_indices.append([0, 0, self.var_dict[tempwstr],self.var_dict[tempsstr]])
                self.poly_coefs.append(2*self.r)

        #penalties for H
        for i in range(self.k):
            for j in range(self.m):
                #+h^2 * r
                temphstr = "h_"+str(i)+str(",")+str(j)
                self.poly_indices.append([0, 0, self.var_dict[temphstr],self.var_dict[temphstr]])
                self.poly_coefs.append(1*self.r)
                #+s_h^2 * r
                tempsstr= 's_h_'+ str(i) + "," + str(j)
                self.poly_indices.append([0, 0, self.var_dict[tempsstr],self.var_dict[tempsstr]])
                self.poly_coefs.append(1*self.r)
                #-2h * r * p
                self.poly_indices.append([0, 0, 0,self.var_dict[temphstr]])
                self.poly_coefs.append(-2*self.r*self.p)
                #-2s_h * r * p
                self.poly_indices.append([0, 0,0,self.var_dict[tempsstr]])
                self.poly_coefs.append(-2*self.r*self.p)
                #+2hs_h * r
                self.poly_indices.append([0, 0, self.var_dict[temphstr],self.var_dict[tempsstr]])
                self.poly_coefs.append(2*self.r)

    #Regularization not used in paper (could be used in a Future Work)
    def encode_regularization(self):
    #Implements regularization based on the Frobenius norm
        #Going over W
        for i in range(self.n):
            for j in range(self.k):
                tempwstr = "w_"+str(i)+str(",")+str(j)
                #Check if w^2 exists, this will happen if wh_constraints were True
                if self.wh_constraint == False:
                    #Add it as a new entry in the lists
                    self.poly_indices.append([0, 0, self.var_dict[tempwstr],self.var_dict[tempwstr]])
                    self.poly_coefs.append(self.alpha_W)
                else:
                    #Modify current entrys in the lists
                    temp_idx = self.poly_indices.index([0, 0, self.var_dict[tempwstr],self.var_dict[tempwstr]])#Get index of w^2 
                    #Use temp_idex to locate the coefficient and add to it
                    self.poly_coefs[temp_idx] += self.alpha_W
            
        #Going over H
        for i in range(self.k):
            for j in range(self.m):
                temphstr = "h_"+str(i)+str(",")+str(j)
                #Check if w^2 exists, this will happen if wh_constraints were True
                if self.wh_constraint == False:
                    #Add it as a new entry in the lists
                    self.poly_indices.append([0, 0, self.var_dict[temphstr],self.var_dict[temphstr]])
                    self.poly_coefs.append(self.alpha_H)
                else:
                    #Modify current entrys in the lists
                    temp_idx = self.poly_indices.index([0, 0, self.var_dict[temphstr],self.var_dict[temphstr]])#Get index of w^2 
                    #Use temp_idex to locate the coefficient and add to it
                    self.poly_coefs[temp_idx] += self.alpha_H



    
    #Establish connection to QCI
    def establish_connection(self,url,api_token):
        self.url = url
        self.api_token = api_token
        self.client = QciClient(url=self.url, api_token=self.api_token) #<--- We will use client in other functions

    #Create the problem in the format QCI wants for Dirac-3 (for continuous values)
    def run_problem(self,relaxation_schedule=1,solution_precision=None,num_samples=1):
        self.solution_precision = solution_precision #Make this into a instance variable
        #---Make it into a file format Dirac-3 understands
        self.data = [{"idx": idx, "val": val} for idx, val in zip(self.poly_indices, self.poly_coefs)]

        self.file = {
            "file_name": "dirac_3_continuous_variable_example",
            "file_config": {
                "polynomial": {
                    "num_variables": len(self.var_dict),
                    "min_degree": 1,
                    "max_degree": 4,
                    "data": self.data,
                }
            }
        }

        self.file_response = self.client.upload_file(file=self.file)

        #---Creating a job
        #setting job_params
        #(Common job params) first
        job_params = {}
        job_params['device_type'] = 'dirac-3'
        job_params['relaxation_schedule'] = relaxation_schedule
        job_params['sum_constraint'] = self.sum_constraint
        job_params['num_samples'] = num_samples
        if solution_precision != None:
            job_params['solution_precision']=solution_precision
        
        #Build job body
        self.job_body = self.client.build_job_body(
            job_type='sample-hamiltonian',
            job_name='test_continuous_variable_hamiltonian_job', # user-defined string, optional
            job_params=job_params,
            polynomial_file_id=self.file_response['file_id']
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
    
    #run the problem on Dirac-3 but this time with integers in range 0 to a max_int
    def run_problem_integer(self,max_int = 7, relaxation_schedule=1,num_samples=1):
        self.solution_precision = None
        num_levels = []
        for i in range(len(self.var_dict)):
            num_levels.append(max_int + 1)

        #PREPARE FILE OBJECT
        data_int_problem = [{"idx": idx, "val": val} for idx, val in zip(self.poly_indices, self.poly_coefs)]
        file_int_problem = {
        "file_name": "dirac_3_integer_example",
        "file_config": {
        "polynomial": {
            "num_variables": len(self.var_dict),
            "min_degree": 1,
            "max_degree": 4,
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

        
    #Gets W and H from the results
    def get_result(self):

        if self.solution_precision == None:
            #Get undistilled solution
            solution = self.job_response['results']['solutions'][0] #extract answer
        else:
            #Get distilled solution
            solution = self.job_response['results']['distilled_solutions'][0] #extract answer

        #Get W
        for i in range(self.n):
            for j in range(self.k):
                tempwstr = 'w_'+str(i)+","+str(j)
                self.W[i,j] = solution[self.var_dict[tempwstr]-1]

        #Get H
        for i in range(self.k):
            for j in range(self.m):
                temphstr = 'h_'+str(i)+","+str(j)
                self.H[i,j] = solution[self.var_dict[temphstr]-1]

        #Get V_rec
        self.V_rec = self.W @ self.H
    

    def calculate_energy_matrix(self,W,H):
        #Calculate energy from some W and H matrices
        sol_array = [0.0] *(W.size + H.size + 1)
        #Get values from W
        for i in range(self.n):
            for j in range(self.k):
                tempwstr = 'w_'+str(i)+","+str(j)
                sol_array[self.var_dict[tempwstr]-1] = W[i,j]
        
        #Get values from H
        for i in range(self.k):
            for j in range(self.m):
                temphstr = 'h_'+str(i)+","+str(j)
                sol_array[self.var_dict[temphstr]-1] = H[i,j]
                
        
        #Add some slack
        sol_array[-1] = 0.9

        #Now let us go over the formulation
        energy = 0.0
        for i in range(len(self.poly_coefs)):
            tempenergy = self.poly_coefs[i]
            for item in self.poly_indices[i]:
                if item !=0: # That means there is a genuine variable
                    tempenergy *= sol_array[item-1] #0 based index
            
            energy += tempenergy
        
        return energy


                







        

            
