from plotting import plot_cost, plot_fidelities_for_dict, plot_histogram, plot_cost_batch_individual
import numpy as np
from setup_file import load_dictionary
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt


N_INIT = 5
ALLOCATED_BUDGET = 50

fileName = 'SampleSpaces/20250304-145435.csv'
#dict_filename = 'SearchDictionaries/20250304-111416'
dict_filename = 'SearchDictionaries/20250304-145550'
batch_dict_filename = 'SearchDictionaries/Batch_20250304-150046'



modelDict = load_dictionary(f'{dict_filename}')

for key, value in modelDict.items():
    print(f"Key: {key}, Value: {value}")

domain = np.loadtxt(f'{fileName}', delimiter=',')
searchDictBatch = load_dictionary(f'{batch_dict_filename}')
'''
#plots
plt.figure()
plot_fidelities_for_dict(modelDict, domain)
plt.show()

plt.figure()
plot_cost(domain, modelDict,f'Initialisation {N_INIT} and {ALLOCATED_BUDGET} Budget' )
plt.show()

plt.figure()
plot_histogram(domain)
plt.show()

plt.figure()
plot_cost_batch_individual(domain, searchDictBatch)
plt.show()
'''