from setup_file import *


#MES, TVR and single-fidelity EI functions
#Also code to run optimisation cycle & batch processes

def runMes(model, bounds, fidelity_history, previous_evaluations=None, train_x_past=None):
    fidelities = np.unique(fidelity_history)

    min_vals = bounds[0]  
    max_vals = bounds[1]

    candidate_set_no_hf = min_vals + (max_vals - min_vals) * torch.rand(10000, bounds.shape[1])
    
    candidate_set = torch.tensor(np.concatenate((candidate_set_no_hf, np.array([[random.choice(fidelities) for x in range(10000)]]).T), axis=1))

    fidelity_index = candidate_set.shape[1] -1
    target_fidelities = {fidelity_index: 1.0} 
    print('runMES target_fidelities', target_fidelities)
    
    cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=1.0) 
    #cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=0.1)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    acquisition = qMultiFidelityMaxValueEntropy(
            model=model,
            cost_aware_utility=cost_aware_utility,
            project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
            candidate_set=candidate_set,
            maximize=True
        )
    
    acquisitionScores = acquisition.forward(candidate_set.reshape(-1, 1, bounds.shape[1]+1))

    return acquisitionScores, candidate_set


#un-modified version (see modified version in fixing.ipynb)
def runEI(model, Xrpr, previous_evaluations, train_x_past=None): #how is this handling discrete variables
    Xrpr = torch.tensor(Xrpr)
    acquisition = ExpectedImprovement(
            model=model,
            best_f= max(previous_evaluations)
        )
    
    acquisitionScores =  acquisition.forward(Xrpr.reshape(-1,1, Xrpr.shape[1]) ).detach() 
    return acquisitionScores

'''
def runTVR(model, Xrpr, previous_evaluations=None, train_x_past=None):
    Xrpr_hf = Xrpr[np.where(Xrpr[:, -1]==1)]
    # indices = np.where(train_x_past[:, 1] == 1)

    acquisition_scores = runEI(model, Xrpr_hf, previous_evaluations)
    max_hf_ind = acquisition_scores.argmax()

    index_in_xrpr = Xrpr.tolist().index(Xrpr_hf[max_hf_ind].tolist())
    Xrpr = torch.tensor(Xrpr)

    posterior = model.posterior(Xrpr)

    pcov = posterior.distribution.covariance_matrix
    p_var = posterior.variance
    hf_max_cov = pcov[index_in_xrpr]
    hf_max_var = hf_max_cov[index_in_xrpr]
    cost = Xrpr[:, -1]
    
    return hf_max_cov ** 2 / (p_var.reshape(-1) * hf_max_var * cost)   
    
# This approach transforms the fidelity column to be as described in the paper, i.e. [1] -> [0] and [0.1] -> 1
# We do this transformation repeatedly as we wish to keep the data as it is, since the output from these searches 
# are required to stick to a specific format so that the graphing functionality knows how to deal with it.
def runTVR_mod(model, Xrpr, previous_evaluations=None, train_x_past=None):
    X_rpr_transf = copy.deepcopy(Xrpr)

    for row in range(len(X_rpr_transf)):
        X_rpr_transf[row][-1] = 1 if X_rpr_transf[row][-1] != 1 else 0

    #Get hf data-points. 
    Xrpr_hf = X_rpr_transf[np.where(X_rpr_transf[:, -1]==0)]

    acquisition_scores = runEI(model, Xrpr_hf, previous_evaluations)
    max_hf_ind = acquisition_scores.argmax()

    index_in_xrpr = X_rpr_transf.tolist().index(Xrpr_hf[max_hf_ind].tolist())
    Xrpr_transf = torch.tensor(X_rpr_transf)

    posterior = model.posterior(Xrpr_transf)

    pcov = posterior.distribution.covariance_matrix
    p_var = posterior.variance
    hf_max_cov = pcov[index_in_xrpr]
    hf_max_var = hf_max_cov[index_in_xrpr]
    cost = Xrpr[:, -1]
    return  hf_max_cov ** 2 / (p_var.reshape(-1) * hf_max_var * torch.tensor(cost))   

'''

def optimiseAcquisitionFunction(acq_function, index_store, bounds, candidate_set):
    """Suggest a new candidate within the search space using a given acquisition function."""
    
    print('acq_func', acq_function)
    candidate_idx = torch.argmax(acq_function)
    print('candidate_idx', candidate_idx)
    candidate = candidate_set[candidate_idx]
    print('candidate', candidate)
    candidate_tensor = torch.tensor(candidate[0:-1], dtype=torch.float32).unsqueeze(0)
    fidelity = candidate[-1].item()
    

    candidate_tuple = tuple(candidate.squeeze().tolist())
    if candidate_tuple in index_store:
        print("Candidate already evaluated, selecting a new one.")
        return optimiseAcquisitionFunction(acq_function, index_store, bounds, candidate_set)

    index_store.append(candidate_tuple)  
    print("candidate_tensor", candidate_tensor)
    # Evaluate the new candidate
    column_names = ["Catalyst Loading", "Temperature", "Residence Time"]
    output = evaluate_tensor(candidate_tensor, column_names=column_names)

    if math.isclose(fidelity, 0.1, rel_tol=1e-5): #lf case -> add noise
        output = output + random.gauss(0, 10)

    
    return candidate, output, fidelity

def run_entire_cycle(train_x_full, 
                     train_obj, 
                     fidelity_history, 
                     index_store, 
                     func,
                     sf=False, 
                     no_of_iterations=100000, 
                     allocated_budget=100000
                     ):
    """Runs the full optimization cycle with continuous search space handling."""
    
    train_x_full = copy.deepcopy(train_x_full)
    train_obj = copy.deepcopy(train_obj)
    fidelity_history = copy.deepcopy(fidelity_history)
    index_store = copy.deepcopy(index_store)

    budget_sum = sum(fidelity_history)
    iteration_counter = 0
    
    while budget_sum <= allocated_budget - 1 and iteration_counter < no_of_iterations:
        model = SingleTaskGP(train_x_full, train_obj) if sf else SingleTaskMultiFidelityGP(train_x_full, train_obj, data_fidelities=[-1])
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        acquisition_function, candidate_set = func(model=model, bounds = bounds, fidelity_history = fidelity_history, previous_evaluations=train_obj, train_x_past=train_x_full)
        
        top_candidate, evaluation, fidelity = optimiseAcquisitionFunction(acquisition_function, index_store, bounds=bounds, candidate_set = candidate_set)

        print('fidelity',fidelity)
        fidelity_history.append(fidelity)

        train_x_full = torch.cat([train_x_full, top_candidate.unsqueeze(0)])
        train_obj = torch.cat([train_obj, torch.tensor([evaluation]).unsqueeze(-1)])

        iteration_counter += 1
        budget_sum += fidelity
        
    cumulative_cost = np.cumsum(fidelity_history).tolist()
    
    return train_x_full, train_obj, cumulative_cost, index_store

#not yet modified
def run_entire_cycle_batch(experiments, 
                           domain_input, 
                           initial_sample_size, 
                           func, 
                           no_of_iterations=10000, 
                           allocated_budget=10000,
                           predefined_indices_batch=None, 
                           sf=False, 
                           file = True):
    train_x_full_batch = []
    train_obj_batch = []
    cumulative_cost_batch = [] 
    index_store_batch = []
    for j in range(experiments):
        predefined_indices = None if predefined_indices_batch is None else predefined_indices_batch[j]
        train_x_full, train_obj, domain, index_store, fidelity_history = setUpInitialData(domain_input, 
                                                                                        initial_sample_size,
                                                                                        predefined_indices=predefined_indices, 
                                                                                           sf=sf, 
                                                                                           file=file)
        train_x_full, train_obj, cumulative_cost, index_store = run_entire_cycle(
            train_x_full, 
            train_obj, 
            domain, 
            fidelity_history, 
            index_store, 
            func,
            sf=sf,
            no_of_iterations=no_of_iterations,
            allocated_budget=allocated_budget)       
  
        train_x_full_batch.append(train_x_full)
        train_obj_batch.append(train_obj)
        cumulative_cost_batch.append(cumulative_cost)
        index_store_batch.append(index_store)
    return train_x_full_batch, train_obj_batch, cumulative_cost_batch, index_store_batch