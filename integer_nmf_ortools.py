#Houses the function that does integer NMF using google's ortools (CP-SAT)
import numpy as np
from ortools.sat.python import cp_model
import time

#Important : This assumes that the elements of W and H matrices have the range of 0 to 7
#Code made using Claude 4.0, and then making some additions/changes to it 
#Note: n_components is equivalent to the index k (p in the paper) we use for num of col of W
def integer_nmf_ortools(V, n_components, max_iter=1, max_time=30.0, W_init=None, H_init=None, 
                       init=False, value_range=(0, 7), objective='l1', verbose=False,seed=None,threads=None,nzconstraint=False):
    """
    Integer NMF using OR-Tools CP-SAT solver
    
    Parameters:
    -----------
    V : numpy.ndarray
        Input matrix to factorize
    n_components : int
        Number of components (rank)
    max_iter : int
        Maximum iterations (kept for compatibility, not used in CP-SAT)
    max_time : float
        Time limit in seconds
    W_init : numpy.ndarray, optional
        Initial W matrix for warm start
    H_init : numpy.ndarray, optional
        Initial H matrix for warm start
    init : bool
        Whether to use warm start
    value_range : tuple
        Range of integer values (min, max)
    objective : str
        'l1' for L1 norm, 'l2' for squared L2 norm
    verbose : bool
        Whether to print solver information
    
    Returns:
    --------
    W_result : numpy.ndarray
        Factorized W matrix
    H_result : numpy.ndarray
        Factorized H matrix
    info : dict
        Solver information
    """
    m, n = V.shape
    min_val, max_val = value_range
    max_product = max_val * max_val
    max_reconstruction = max_product * n_components
    
    # Create the model
    model = cp_model.CpModel()
    
    # Create variables: W[i,k] and H[k,j] all integers in specified range
    W = {}
    H = {}
    
    for i in range(m):
        for k in range(n_components):
            W[i, k] = model.NewIntVar(min_val, max_val, f'W_{i}_{k}')
    
    for k in range(n_components):
        for j in range(n):
            H[k, j] = model.NewIntVar(min_val, max_val, f'H_{k}_{j}')
    
    # Create auxiliary variables for the products W[i,k] * H[k,j]
    reconstruction = {}
    
    for i in range(m):
        for j in range(n):
            # Sum over k: W[i,k] * H[k,j]
            terms = []
            for k in range(n_components):
                # Create product variable
                product = model.NewIntVar(min_val * min_val, max_product, f'prod_{i}_{k}_{j}')
                model.AddMultiplicationEquality(product, [W[i, k], H[k, j]])
                terms.append(product)
            
            # Sum of products
            reconstruction[i, j] = model.NewIntVar(
                min_val * min_val * n_components, 
                max_reconstruction, 
                f'recon_{i}_{j}'
            )
            model.Add(reconstruction[i, j] == sum(terms))
    
    
    error_vars = []
    if ( 2**(31) - 1 ) > max_reconstruction ** 2:
        V_int = V.astype(int)  # Ensure V is integer
    else:
        V_int = V.astype(np.int64)
    
    # Choose objective function
    if objective == 'l1':
        # L1 reconstruction error (sum of absolute differences)
        for i in range(m):
            for j in range(n):
                error_pos = model.NewIntVar(0, max_reconstruction + abs(V_int[i, j]), f'error_pos_{i}_{j}')
                error_neg = model.NewIntVar(0, max_reconstruction + abs(V_int[i, j]), f'error_neg_{i}_{j}')
                
                model.Add(reconstruction[i, j] - V_int[i, j] == error_pos - error_neg)
                
                total_error = model.NewIntVar(0, max_reconstruction + abs(V_int[i, j]), f'total_error_{i}_{j}')
                model.AddMaxEquality(total_error, [error_pos, error_neg])
                error_vars.append(total_error)
    
    elif objective == 'l2':
        # Squared L2 reconstruction error
        
        for i in range(m):
            for j in range(n):
                diff = model.NewIntVar(
                    -max_reconstruction - abs(V_int[i, j]), 
                    max_reconstruction + abs(V_int[i, j]), 
                    f'diff_{i}_{j}'
                )
                model.Add(diff == reconstruction[i, j] - V_int[i, j])
                squared_error = model.NewIntVar(0, (max_reconstruction + abs(V_int[i, j])) ** 2, f'sq_error_{i}_{j}')
                model.AddMultiplicationEquality(squared_error, [diff, diff])
                error_vars.append(squared_error)
    
    # Objective: minimize total error
    total_error = model.NewIntVar(0, sum([var.Proto().domain[1] for var in error_vars]), 'total')
    model.Add(total_error == sum(error_vars))
    model.Minimize(total_error)
    
    # Warm start
    if init and W_init is not None and H_init is not None:
        W_init_int = np.clip(W_init.astype(int), min_val, max_val)
        H_init_int = np.clip(H_init.astype(int), min_val, max_val)
        
        for i in range(min(W_init_int.shape[0], m)):
            for j in range(min(W_init_int.shape[1], n_components)):
                model.AddHint(W[i, j], W_init_int[i, j])
        
        for i in range(min(H_init_int.shape[0], n_components)):
            for j in range(min(H_init_int.shape[1], n)):
                model.AddHint(H[i, j], H_init_int[i, j])
    
    # Additional constraints (optional)
    # Prevent trivial solution where all values are 0
    if nzconstraint == True:
        non_zero_constraint = []
        for i in range(m):
            for k in range(n_components):
                non_zero_constraint.append(W[i, k])
        for k in range(n_components):
            for j in range(n):
                non_zero_constraint.append(H[k, j])
        
        # At least one non-zero value
        model.Add(sum(non_zero_constraint) >= 1)
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    if threads != None: #set number of threads if given
        solver.parameters.num_search_workers = threads  # Use multiple threads
    if seed != None: #set the seed (if given)
        solver.parameters.random_seed = seed
    start_time = time.time()
    status = solver.Solve(model) #<---Actually solve
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"Solver status: {solver.StatusName(status)}")
        print(f"Solve time: {solve_time:.2f} seconds")
        print(f"Number of search workers: {solver.parameters.num_search_workers}")
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Objective value: {solver.ObjectiveValue()}")
    
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
        
        # Calculate final reconstruction error
        reconstruction_matrix = W_result @ H_result
        if objective == 'l1':
            final_error = np.sum(np.abs(V - reconstruction_matrix))
        else:
            final_error = np.sum((V - reconstruction_matrix) ** 2)
        
        info = {
            'status': solver.StatusName(status),
            'solve_time': solve_time,
            'objective_value': solver.ObjectiveValue(),
            'final_error': final_error,
            'n_variables': model.Proto().variables,
            'n_constraints': len(model.Proto().constraints)
        }
        
        return W_result, H_result, info
    else:
        if verbose:
            print("No solution found")
        
        info = {
            'status': solver.StatusName(status),
            'solve_time': solve_time,
            'objective_value': None,
            'final_error': None,
            'n_variables': model.Proto().variables,
            'n_constraints': len(model.Proto().constraints)
        }
        
        return None, None, info