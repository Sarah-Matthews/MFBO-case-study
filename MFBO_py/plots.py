from plotting import plot_cost, plot_cost_all_lf,plot_fidelities_for_dict, plot_histogram, plot_cost_batch_individual, plot_cost_batch, plot_cost_batch_average, plot_cost_batch_average_simple, plot_cost_batch_individual_old
import numpy as np
from setup_file import load_dictionary
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import pandas as pd


N_INIT = 5
ALLOCATED_BUDGET = 40

#kinda working 0.1, 8 var
#fileName = 'SampleSpaces/20250404-165239.csv'
#dict_filename = 'SearchDictionaries/20250402-112039'

#quite promising - 0.01 cost & 50 runs, subtracting gauss noise
#fileName = 'SampleSpaces/20250404-193456.csv'
#dict_filename = 'SearchDictionaries/20250404-201715'

fileName = 'SampleSpaces/20250405-214710.csv'
dict_filename = 'SearchDictionaries/20250405-191631'


batch_dict_filename = 'SearchDictionaries/Batch_20250405-072219'


modelDict = load_dictionary(f'{dict_filename}')

for key, value in modelDict.items():
    print(f"Key: {key}, Value: {value}")

domain = np.loadtxt(f'{fileName}', delimiter=',')
searchDictBatch = load_dictionary(f'{batch_dict_filename}')


def plot_sample_space(setup_file):
    # Read the sample space data from the CSV file
    domain = pd.read_csv(setup_file, header=None).values

    hf_outputs = []
    lf_outputs = []

    # Iterate through the domain and extract HF and LF outputs
    for i in range(0, len(domain), 2):  # Assuming HF and LF alternate, with HF first
        hf_outputs.append(domain[i, -1])  # Last column as HF output
        lf_outputs.append(domain[i+1, -1])  # Last column of LF (next point)

    paired_outputs = sorted(zip(hf_outputs, lf_outputs), key=lambda x: x[0])
    hf_sorted, lf_sorted = zip(*paired_outputs)

    # Plot HF and LF values
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(hf_sorted) + 1), hf_sorted, label="High Fidelity (HF)", color='blue')
    plt.plot(range(1, len(lf_sorted) + 1), lf_sorted, label="Low Fidelity (LF)", color='red')

    plt.xlabel('Index in Search Space')
    plt.ylabel('Output Value')
    plt.title('High Fidelity vs Low Fidelity Outputs')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()



plot_sample_space(fileName)
#plots
#plt.figure()
#plot_fidelities_for_dict(modelDict, domain)
#plt.show()

filtered_modelDict = {k: v for k, v in modelDict.items() if k in ["MF-MES", "SF-EI", "SF-MES"]}
plt.figure()
plot_fidelities_for_dict(filtered_modelDict, domain, allocated_budget=ALLOCATED_BUDGET, number_init=N_INIT)
plt.show()

plt.figure()
plot_cost(domain, modelDict,f'Initialisation {N_INIT} and {ALLOCATED_BUDGET} Budget' )
plt.show()

plt.figure()
plot_cost_all_lf(domain, modelDict,f'Initialisation {N_INIT} and {ALLOCATED_BUDGET} Budget' )
plt.show()

plt.figure()
plot_histogram(domain)
plt.show()

plt.figure()
plot_cost_batch_individual_old(domain, searchDictBatch)
plt.show()

plt.figure()
plot_cost_batch_individual(domain, searchDictBatch)
plt.show()

#plt.figure()
#plot_cost_batch_average(domain, searchDictBatch, num_points=100)
#plt.show()



plt.figure()
plot_cost_batch_average_simple(domain, searchDictBatch,cost_step=0.05,max_budget=40)
plt.show()

'''
plt.figure()
plot_cost_batch(domain,modelDict,f'Initialisation {N_INIT} and {ALLOCATED_BUDGET} Budget',ALLOCATED_BUDGET,lf=0.01)
plt.show()
'''