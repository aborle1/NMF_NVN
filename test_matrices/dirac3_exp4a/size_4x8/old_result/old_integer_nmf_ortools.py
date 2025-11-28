#Houses the function that does integer NMF using google's ortools (CP-SAT)
import numpy as np
from ortools.sat.python import cp_model

#Important : This assumes that the elements of W and H matrices have the range of 0 to 7
#Code made using Claude 4.0, and then making some additions/changes to it 
#Note: n_components is equivalent to the index k we use for num of col of W
def integer_nmf_ortools(V, n_components, max_iter=1,max_time =30.0,W_init =None, H_init = None, init=False,seed=None,threads=None):
    """
    Integer NMF using OR-Tools CP-SAT solver
    """
    m, n = V.shape

    # Create the model
    model = cp_model.CpModel()

    # Create variables: W[i,k] and H[k,j] all integers 0-7
    W = {}
    H = {}

    for i in range(m):
        for k in range(n_components):
            W[i, k] = model.NewIntVar(0, 7, f'W_{i}_{k}')

    for k in range(n_components):
        for j in range(n):
            H[k, j] = model.NewIntVar(0, 7, f'H_{k}_{j}')

    # Create auxiliary variables for the products W[i,k] * H[k,j]
    products = {}
    reconstruction = {}

    for i in range(m):
        for j in range(n):
            # Sum over k: W[i,k] * H[k,j]
            terms = []
            for k in range(n_components):
                # Create product variable
                product = model.NewIntVar(0, 49, f'prod_{i}_{k}_{j}')  # max is 7*7=49
                model.AddMultiplicationEquality(product, [W[i, k], H[k, j]])
                terms.append(product)

            # Sum of products
            reconstruction[i, j] = model.NewIntVar(0, 49 * n_components, f'recon_{i}_{j}')
            model.Add(reconstruction[i, j] == sum(terms))

    # Minimize reconstruction error (sum of absolute differences)
    error_vars = []
    for i in range(m):
        for j in range(n):
            error_pos = model.NewIntVar(0, max(49 * n_components, 7), f'error_pos_{i}_{j}')
            error_neg = model.NewIntVar(0, max(49 * n_components, 7), f'error_neg_{i}_{j}')

            model.Add(reconstruction[i, j] - int(V[i, j]) == error_pos - error_neg)

            total_error = model.NewIntVar(0, max(49 * n_components, 7), f'total_error_{i}_{j}')
            model.AddMaxEquality(total_error, [error_pos, error_neg])
            error_vars.append(total_error)

    # Objective: minimize total error
    total_error = model.NewIntVar(0, sum([max(49 * n_components, 7)] * len(error_vars)), 'total')
    model.Add(total_error == sum(error_vars))
    model.Minimize(total_error)

    #Warm start
    if init == True:
        for i in range(W_init.shape[0]):
            for j in range(W_init.shape[1]):
                model.AddHint(W[i,j],W_init[i,j])
        
        for i in range(H_init.shape[0]):
            for j in range(H_init.shape[1]):
                model.AddHint(H[i,j],H_init[i,j])

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time  # Time limit
    if seed != None: # Set a seed value
        solver.parameters.random_seed = seed
    if threads != None: #If we want to set a particular number of threads
        solver.parameters.num_search_workers = threads
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract solution
        W_result = np.zeros((m, n_components), dtype=int)
        H_result = np.zeros((n_components, n), dtype=int)

        for i in range(m):
            for k in range(n_components):
                W_result[i, k] = solver.Value(W[i, k])

        for k in range(n_components):
            for j in range(n):
                H_result[k, j] = solver.Value(H[k, j])

        return W_result, H_result
    else:
        print("No solution found")
        return None, None
