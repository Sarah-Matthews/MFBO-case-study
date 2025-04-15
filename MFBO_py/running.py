from setup_file import *
from acqfs import *
from plotting import *

plt.ion()  #turn on interactive mode


ALLOCATED_BUDGET_BATCH = 20

setup_file = setUpSampleSpace(spaceSize=100, var=12, lf_cost=0.01)
print('setup file:',setup_file)

N_INIT = 5
ALLOCATED_BUDGET = 50
fileName = setup_file
print(fileName)
train_x_full, train_obj, domain, index_store, fidelity_history = setUpInitialData(fileName, N_INIT)

print('Correlation (HF vs LF):', compute_correlation(domain=domain))
'''

train_x_full_tvr, train_obj_tvr, cumulative_cost_tvr, index_store_tvr = run_entire_cycle(
    train_x_full, 
    train_obj, 
    domain, 
    fidelity_history,
    index_store,
    func=runTVR,
    allocated_budget=ALLOCATED_BUDGET,
    )
'''
'''
print(domain)
train_x_full_sf_lf, train_obj_sf_lf, domain_sf_lf, index_store_sf_lf, fidelity_history_sf_lf = convertMFDatatoSFData_LF(domain, index_store)
print(index_store_sf_lf)
print(fidelity_history_sf_lf)
train_x_full_ei_lf, train_obj_ei_lf, cumulative_cost_ei_lf, index_store_ei_lf= run_entire_cycle(
    train_x_full_sf_lf, 
    train_obj_sf_lf, 
    fidelity_history_sf_lf,
    index_store_sf_lf,
    runEI_lf,
    sf=True,
    allocated_budget=ALLOCATED_BUDGET)

'''

train_x_full_mes, train_obj_mes, cumulative_cost_mes, index_store_mes = run_entire_cycle(
    train_x_full, 
    train_obj, 
    fidelity_history,
    index_store,
    runMes,
    allocated_budget=ALLOCATED_BUDGET
    )

# The single fidelity case requires different initial data. For example, the initial sample must be all HF and the domain is all HF.
# The fidelity history is less important here as we know all chosen points will be HF, but we keep them so that we can reuse the same function.
train_x_full_sf, train_obj_sf, domain_sf, index_store_sf, fidelity_history_sf = convertMFDatatoSFData(domain, index_store)
print(index_store_sf)
print(fidelity_history_sf)
train_x_full_ei, train_obj_ei, cumulative_cost_ei, index_store_ei= run_entire_cycle(
    train_x_full_sf, 
    train_obj_sf, 
    fidelity_history_sf,
    index_store_sf,
    runEI,
    sf=True,
    allocated_budget=ALLOCATED_BUDGET)

train_x_full_sfmes, train_obj_sfmes, domain_sfmes, index_store_sfmes, fidelity_history_sfmes = convertMFDatatoSFData(domain, index_store)
print(index_store_sfmes)
print(fidelity_history_sfmes)
train_x_full_sfmes, train_obj_sfmes, cumulative_cost_sfmes, index_store_sfmes= run_entire_cycle(
    train_x_full_sfmes, 
    train_obj_sfmes, 
    fidelity_history_sfmes,
    index_store_sfmes,
    runSF_MES,
    sf=True,
    allocated_budget=ALLOCATED_BUDGET)



modelDict = {
    
    "MF-MES": (train_x_full_mes, train_obj_mes, cumulative_cost_mes),
    #"MF-TVR": (train_x_full_tvr, train_obj_tvr, cumulative_cost_tvr),
    "SF-EI" : (train_x_full_ei, train_obj_ei, cumulative_cost_ei),
    #"SF-EI-LF" : (train_x_full_ei_lf, train_obj_ei_lf, cumulative_cost_ei_lf),
    "SF-MES" : (train_x_full_sfmes, train_obj_sfmes, cumulative_cost_sfmes),
             }
dict_filename = save_dictionary(modelDict, batch=False) 
print(dict_filename)
print('setup file:',setup_file)
modelDict = load_dictionary(f'{dict_filename}')
domain = np.loadtxt(f'{fileName}', delimiter=',')



#batch section

N_EXP_BATCH = 5
INIT_SAMPLE_SIZE_BATCH=5
ALLOCATED_BUDGET_BATCH = 50

predefined_indices = generate_batch_indices(domain, INIT_SAMPLE_SIZE_BATCH, N_EXP_BATCH)
  

train_x_full_mes_batch, train_obj_mes_batch, cumulative_cost_mes_batch, index_store_mes_batch = run_entire_cycle_batch(
    N_EXP_BATCH,
    fileName,
    INIT_SAMPLE_SIZE_BATCH, 
    runMes,
    allocated_budget = ALLOCATED_BUDGET_BATCH,
    predefined_indices_batch = predefined_indices
    );

train_x_full_ei_batch, train_obj_ei_batch, cumulative_cost_ei_batch, index_store_ei_batch = run_entire_cycle_batch(
    N_EXP_BATCH,
    fileName,
    INIT_SAMPLE_SIZE_BATCH, 
    runEI,
    allocated_budget = ALLOCATED_BUDGET_BATCH,
    sf=True,
    predefined_indices_batch = predefined_indices);

train_x_full_sfmes_batch, train_obj_sfmes_batch, cumulative_cost_sfmes_batch, index_store_sfmes_batch = run_entire_cycle_batch(
    N_EXP_BATCH,
    fileName,
    INIT_SAMPLE_SIZE_BATCH, 
    runSF_MES,
    allocated_budget = ALLOCATED_BUDGET_BATCH,
    sf=True,
    predefined_indices_batch = predefined_indices);

searchDictBatch = {
    
    "MF-MES": (train_x_full_mes_batch, train_obj_mes_batch, cumulative_cost_mes_batch),

    "SF-EI": (train_x_full_ei_batch, train_obj_ei_batch, cumulative_cost_ei_batch),

    "SF-MES": (train_x_full_sfmes_batch, train_obj_sfmes_batch, cumulative_cost_sfmes_batch),
                   }
batch_dict_filename = save_dictionary(searchDictBatch, batch=True)

searchDictBatch = load_dictionary(f'{batch_dict_filename}')


