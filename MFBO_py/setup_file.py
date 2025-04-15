import pandas as pd
import numpy as np
import seaborn as sns
import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP, SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP #should be using this due to discrete variable
from botorch.posteriors.gpytorch import scalarize_posterior
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition import PosteriorMean 
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch import test_functions
import numpy as np
from scipy.spatial.distance import cdist
from botorch import fit_gpytorch_mll
torch.set_printoptions(precision=12, sci_mode=False)
import copy
import math
import matplotlib.pyplot as plt
import random
import time
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import summit
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from summit.utils.dataset import DataSet
from summit import *
import pkg_resources, pathlib


#Summit and search space setup

emulator = get_pretrained_reizman_suzuki_emulator(case=1)

parameter_space = {
        "catalyst_loading": (0.5, 2.0),
        "temperature": (30, 110), 
        "t_res": (60, 600),
    }
bounds = torch.tensor(
    [[v[0] for v in parameter_space.values()],  # Lower bounds
     [v[1] for v in parameter_space.values()]],  # Upper bounds
    dtype=torch.float32
)

#adapted evaluate candidates that interprets tensors required for BOTorch and converts to df expected by summit

#adapted evaluate candidates function that interprets tensors required for BOTorch and converts to df expected by summit
def evaluate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:

    name_map = {
        "Catalyst Loading": "catalyst_loading",
        "Residence Time": "t_res",
        "Temperature": "temperature",
        "Yield": "yld",
        "TON": "ton",
    }
    candidates = candidates.rename(columns=name_map)
    candidates = candidates.astype(np.float64) 

    if 'catalyst' not in candidates.columns:
        candidates.insert(0, 'catalyst', 'P1-L4') #hard coding catalayst P1-L1 or P1-L4

    conditions = summit.DataSet.from_df(candidates)

    emulator_output = emulator.run_experiments(
        conditions, rtn_std=True
    ).rename(columns=dict(zip(name_map.values(), name_map.keys())))
    
    return emulator_output["Yield"]


def tensor_to_dataframe(tensor_data, column_names):
    """Convert a tensor to a Pandas DataFrame with given column names."""
    return pd.DataFrame(tensor_data.numpy(), columns=column_names)

def evaluate_tensor(train_x_full, column_names):
    """Evaluates each row in a tensor using the evaluate_candidates function."""
    df_candidates = tensor_to_dataframe(train_x_full, column_names)
    

    results = []
    for _, row in df_candidates.iterrows():
        result = evaluate_candidates(pd.DataFrame([row]))  # convert single row to DataFrame
        results.append(result.values[0])  # extract the yield value
    

    return torch.tensor(results).unsqueeze(-1)  




def sample_parameters(spaceSize, parameter_space):
    """
    Generates samples from the defined parameter space.
    """
    sampled_points = []
    
    for _ in range(spaceSize):
        sample = []
        for param, values in parameter_space.items():
            if isinstance(values, tuple): 
                sample.append(random.uniform(*values))
            elif isinstance(values, list): 
                sample.append(random.choice(values))
        sampled_points.append(sample)
    
    return np.array(sampled_points)

def setUpSampleSpace(spaceSize=200, var=8, lf_cost=0.01): #function creates a sample space of random points (both hf and lf) within bounds

    parameter_space = {
        "catalyst_loading": (0.5, 2.0),
        "temperature": (30, 110), 
        "t_res": (60, 600),
    }

    lf_cost = torch.tensor([lf_cost])
    Xpr_before = sample_parameters(spaceSize, parameter_space)
    Xpr = [i.astype(np.float64) for i in Xpr_before]

    Xpr_tensor = torch.tensor(Xpr, dtype=torch.float32)

    column_names = [ "Catalyst Loading", "Temperature", "Residence Time"]
    output =evaluate_tensor(Xpr_tensor, column_names=column_names)

    X_total_hf = torch.cat((Xpr_tensor, output), dim=1)

    ones_column = torch.ones((X_total_hf.shape[0], 1), dtype=torch.float32)

    X_total_hf = torch.cat((X_total_hf[:, :-1], ones_column, X_total_hf[:, -1:]), dim=1) #add 1.0 fidelity to hf tensors

    domain = []
    for index, hf in enumerate(X_total_hf):
        domain.append(hf)

        #lf_y = hf[-1] + torch.tensor(random.gauss(0, var))  #adding gaussian noise
        noise = abs(random.gauss(0, var))  # always positive
        lf_y = hf[-1] - torch.tensor(noise)  # subtract noise i.e. LF never higher than HF

        lf_y = torch.tensor(max(0, lf_y.item())).reshape(1) #prevent negative target values i..e cap at zero

        value = torch.cat(( lf_cost, lf_y), dim=0)
        updated_lf = torch.cat((Xpr_tensor[index][0:3], value), dim=0)
        domain.append(updated_lf)

    #print(domain)
    unique_domain = set(tuple(row.tolist()) for row in domain)
    print(f"Unique samples: {len(unique_domain)}, Total samples: {len(domain)}")    
    domain = torch.stack(domain)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fileName = 'SampleSpaces/' + timestr + '.csv'
    os.makedirs('SampleSpaces/', exist_ok=True)
    np.savetxt(fileName, domain, delimiter=',')

    return fileName

#creates an initial sample from the overall sample space
def setUpInitialData(sampleSpaceName, initialSize=10, predefined_indices = None, sf=False, file=True):
      # The file argument is telling us whether we expect the sampleSpaceName to be a file or the actual domain is already in memory.
      # The predefined_indices argument us used in the batch case across multiple search-algorithms where we want 
      #  each element in the batch to have the same intitial set up so that we can compare the averages fairly.
      sampleSpace = np.loadtxt(sampleSpaceName, delimiter=',') if file else sampleSpaceName
      if predefined_indices is None:
            bad_range = True
            top_size = len(sampleSpace) //20
            hf_points = sampleSpace[np.where(sampleSpace[:, -2]==1)]
            top_5_percent = hf_points[hf_points[:, -1].argsort()[::-1]][0:top_size, 0]

            while bad_range:
                  bad_range = False
                  sampleSpace_hf = sampleSpace[np.where(sampleSpace[:, -2]==1)]
                  size = len(sampleSpace_hf)
                  index_store = random.sample(range(size), initialSize)
                  #This gets the high fidelity and low fidelity points in pairs if we're doing MF.
                  sampleSpace, index_store = (sampleSpace_hf, index_store) if sf else (sampleSpace, [2 * x  for x in  index_store] + [1 + 2 * x for x in index_store])
                  fidelity_history = sampleSpace[index_store, -2]
                  #print(fidelity_history)
                  train_X = sampleSpace[index_store, :-1]
                  train_obj = sampleSpace[index_store, -1:]

                  #Do not have an intitial sample that includes the top 5% of points                  
                  for row in train_X:
                       if row[0] in top_5_percent:
                            bad_range=True
                            break
                  
            return torch.tensor(train_X), torch.tensor(train_obj), sampleSpace, index_store, fidelity_history.flatten().tolist()
      else:
            fidelity_history = sampleSpace[predefined_indices, -2]
            train_X = sampleSpace[predefined_indices, :-1]
            train_obj = sampleSpace[predefined_indices, -1:]
            return torch.tensor(train_X), torch.tensor(train_obj), sampleSpace, predefined_indices, fidelity_history.flatten().tolist()
      


#general helper functions:


def generate_batch_indices(sampleSpaceName, initialSize=5, batch_size=5):
      batch_index_store = []
      for batch_no in range(batch_size):
          _, _, _, index_store,_ = setUpInitialData(sampleSpaceName, initialSize, file=False)
          batch_index_store.append(index_store)
      return batch_index_store
     
def create_dictionary_from_batch(batch_dictionary):
    keys = batch_dictionary.keys()
    expanded_dict={}
    for key in keys:
        trainx_batch, objx_batch, cumx_batch = batch_dictionary[key]
        for i in range(len(trainx_batch)):
            expanded_dict[key+str(i)] = (trainx_batch[i], objx_batch[i], cumx_batch[i])
    return expanded_dict
     
# Required when we want to ensure that the sf has the same hf points in its intitial sampel as the mf case.
def convertMFDatatoSFData(sampleSpace, indexStore):
      sampleSpace_hf = sampleSpace[np.where(sampleSpace[:, -2]==1)]
      index_store = [x // 2 for x in indexStore if x % 2 == 0]
      
      return torch.tensor(sampleSpace_hf[index_store, : -1]), torch.tensor(sampleSpace_hf[index_store, -1:]), sampleSpace_hf, index_store, sampleSpace_hf[index_store, -2].flatten().tolist()#sampleSpace[index_store, 1].flatten().tolist()
    
def convertMFDatatoSFData_LF(sampleSpace, indexStore):
      #sampleSpace_lf = sampleSpace[np.where(sampleSpace[:, -2]==0.1)]
      sampleSpace_lf = sampleSpace[np.isclose(sampleSpace[:, -2], 0.01, atol=1e-6)] #chnage for changing lf_cost

      index_store = [x // 2 for x in indexStore if x % 2 == 0]
      
      return torch.tensor(sampleSpace_lf[index_store, : -1]), torch.tensor(sampleSpace_lf[index_store, -1:]), sampleSpace_lf, index_store, sampleSpace_lf[index_store, -2].flatten().tolist()#sampleSpace[index_store, 1].flatten().tolist()
    

def save_dictionary(dictionary, batch=False, root='SearchDictionaries'):
      os.makedirs(root, exist_ok=True)
      timestr = time.strftime("%Y%m%d-%H%M%S")
      fileName = root + '/' + 'Batch_' + timestr if batch else root + '/' + timestr
      with open(fileName, 'wb') as handle:
         pickle.dump(dictionary, handle)
      return fileName

def load_dictionary(file):
    with open(file, 'rb') as inp:
      output = pickle.load(inp)
      return output

def save_image(fig, root='Images/'):
      os.makedirs(root, exist_ok=True)
      timestr = time.strftime("%Y%m%d-%H%M%S")
      fig.savefig(f'{root}/{timestr}')
            

#def compute_correlation(domain):
      #hf_points = np.where(domain[:, -2] == 1)
      #lf_points = np.where(domain[:, -2] != 1)
      #print(domain[hf_points, -1], domain[lf_points, -1])
      #return np.corrcoef(domain[hf_points, -1], domain[lf_points, -1])[0,1]

def compute_correlation(domain):
    hf_values = []
    lf_values = []
    
    for i in range(0, len(domain), 2):  # Iterate in steps of 2
        hf_values.append(domain[i, -1])  # HF values
        lf_values.append(domain[i+1, -1])  # Corresponding LF values
    
    return np.corrcoef(hf_values, lf_values)[0, 1]
