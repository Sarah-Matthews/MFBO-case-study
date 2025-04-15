from setup_file import *
from acqfs import *
from scipy.interpolate import interp1d
import matplotlib.lines as mlines

#functions for plotting graphs

def plot_fidelities(samples, targets, cumulated_cost, title, total_domain,  colours=['red', 'blue', 'green']):
    max_in_space= np.max(total_domain[np.where(total_domain[:, -2] == 1.0)][:, -1])
    max_reached = targets[np.where(samples[:, -1] == 1.0)].max()
    index_array = np.where(targets == max_reached)[0]
    unique_fidelities = np.unique(np.round(total_domain[:, -2], 3))
    print('unique_fidelities',unique_fidelities)
    for index in index_array:
        if samples[index, -1] == 1.0:
            index_of_max = index 
    #print(index_of_max)
    #index_of_max = index_array[0].item()
    correlation = compute_correlation(domain=total_domain)
    print('correlation', correlation)
    samples = samples.detach().numpy()
    targets = targets.detach().numpy()
    fidelities = list(dict.fromkeys(np.round(samples[:, -1], 3)))
    fidelities.sort(reverse=True)
    print('fidelities in plot_fidelities', fidelities)
    for fidelity in fidelities:
        fidelity_target=[]
        fidelity_iteration=[]
        for i in range(len(cumulated_cost)):
            if (round(samples[i, -1],3) == fidelity):
                fidelity_target.append(targets[i])
                fidelity_iteration.append(cumulated_cost[i])
        # maximum_target.append(max(df_total[df_total['fidelity']==fidelity]['target']))
        legend_text = f'Fidelity: {round(fidelity, 3)}'

        plt.scatter(fidelity_iteration, fidelity_target, label=legend_text, color=colours[fidelities.index(fidelity)], alpha=0.5)
    low_fidelity_value = np.min(unique_fidelities)
    
    #plt.axhline(y=max_in_space, color='black', linestyle='--', label='Domain Optimum')
    plt.axvline(x=cumulated_cost[index_of_max], color='black', linestyle=':')
    plt.axhline(y=max_reached, color='black', linestyle=':', label='Obtained Optimum')
    
    # plt.legend(loc="lower right")
    plt.xlabel("Accumulated Cost")
    plt.ylabel("Evaluation")
    #plt.ylim([min(targets)-1, max_in_space +1])
    
    #if fidelity == low_fidelity_value:
        #plt.title(f'{title} \n LF cost: {low_fidelity_value}, Correlation: {round(correlation, 3)}')
    #else:
    plt.title(title)



def plot_fidelities_for_dict(dictionary, dictionary_domain, allocated_budget=50, number_init=5):
    figure = plt.figure(figsize=(13,11))
    no_of_rows = 1
    no_of_columns = math.ceil(len(dictionary.keys()))
    # for id, key in enumerate(dictionary):
    #     plt.subplot(no_of_rows, 2, id + 1)
    #     plot_fidelities(dictionary[key][0], dictionary[key][1], dictionary[key][2], f'{key}', dictionary_domain)
    for id, key in enumerate(list(dictionary.keys())):
        plt.subplot(no_of_rows, no_of_columns, id + 1)
        plot_fidelities(dictionary[key][0], dictionary[key][1], dictionary[key][2], f'{key}', dictionary_domain)
    handles, labels = plt.gca().get_legend_handles_labels()
    #for key in list(dictionary.keys())[-1:]:
        #plt.subplot(2, 2, 4)
        #plot_fidelities(dictionary[key][0], dictionary[key][1], dictionary[key][2], f'{key}', dictionary_domain)
    # plt.suptitle(f'{number_init} Initial Samples, {allocated_budget} Budget, Domain-size {int(len(dictionary_domain)/2)}',  size=16)
    plt.subplots_adjust(top=0.93)
    legend_entries = []
    legend_entries.append(mlines.Line2D([],[],color='red', marker='o',linestyle='None', markersize=8,label='Fidelity: 1.0', alpha = 0.6))
    legend_entries.append(mlines.Line2D([],[],color='blue', marker='o',linestyle='None', markersize=8,label='Fidelity: 0.01', alpha = 0.6)) #change depending on lf_cost
    legend_entries.append(mlines.Line2D([], [], color='black', linestyle=':', label='Obtained Optimum'))
    #plt.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor = (0, -0.01, 1, 1),
           #bbox_transform = plt.gcf().transFigure)
    plt.legend(handles=legend_entries, loc='lower center', ncol=4, bbox_to_anchor = (0, -0.01, 1, 1),
           bbox_transform = plt.gcf().transFigure)
    figure.subplots_adjust(hspace=0.3)
    plt.show(block=False)

plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14) 
plt.rc('legend', fontsize=16)
plt.rc('axes', titlesize=16)



# Here we plot the maximum high-fidelity target reached so far with a given cost. 
def plot_cost(domain, dictionary, title): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        cumulative_cost_array = np.array(cumulative_cost)
        hf_indices = np.where(train_x_full[:, -1]==1.0)
        high_fidelity_obj = train_obj[np.where(train_x_full[:, -1]==1.0)].detach().numpy()
        accum_target = []
        for i in range(len(high_fidelity_obj)):
            accum_target.append(max(high_fidelity_obj[0:i+1]-5))

        plt.plot(cumulative_cost_array[hf_indices ], accum_target, label=search_alg)    
        #plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
        plt.xlabel('Accumulated Cost')
        plt.ylabel('Max High-Fidelity Evaluation')
        #plt.title(title)
        plt.legend(loc = 'lower right')
        plt.minorticks_on()
        plt.grid(True)
        plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   
        plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
        plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
        plt.show(block=False)

def plot_cost_all_lf(domain, dictionary, title): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        cumulative_cost_array = np.array(cumulative_cost)
        #hf
        hf_indices = np.where(train_x_full[:, -1]==1.0)
        high_fidelity_obj = train_obj[np.where(train_x_full[:, -1]==1.0)].detach().numpy()
        accum_hf = [np.max(high_fidelity_obj[:i+1]) for i in range(len(high_fidelity_obj))]
        # Low-fidelity values
        #lf_indices = np.where(train_x_full[:, -1] == 0.1)[0]
        lf_indices = np.where(np.isclose(train_x_full[:, -1], 0.1, rtol=1e-4, atol=1e-6)) #change this depending on lf_cost!
        low_fidelity_obj = train_obj[lf_indices].detach().numpy()
        accum_lf = [np.max(low_fidelity_obj[:i+1]) for i in range(len(low_fidelity_obj))]
        
        if len(hf_indices[0]) > 0:
            plt.plot(cumulative_cost_array[hf_indices], accum_hf, label=f"{search_alg} (HF)")


        if len(lf_indices[0]) > 0:
            plt.plot(cumulative_cost_array[lf_indices], accum_lf, label=f"{search_alg} (LF)", linestyle="--")

        
    #plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Evaluation')
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.show(block=False)


    
'''
def plot_cost_all(domain, dictionary, title): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        cumulative_cost_array = np.array(cumulative_cost)
        hf_indices = np.where(train_x_full[:, -1]==1.0)
        high_fidelity_obj = train_obj[np.where(train_x_full[:, -1]==1.0)].detach().numpy()
        accum_target = []
        for i in range(len(high_fidelity_obj)):
            accum_target.append(max(high_fidelity_obj[0:i+1]))

        plt.plot(cumulative_cost_array[hf_indices ], accum_target, label=search_alg)    
        plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
        plt.xlabel('Accumulated Cost')
        plt.ylabel('Max High-Fidelity Evaluation')
        plt.title(title)
        plt.legend()
        plt.show(block=False)
'''

def plot_cost_all(domain, dictionary, title): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        cumulative_cost_array = np.array(cumulative_cost)
        #hf
        hf_indices = np.where(train_x_full[:, -1]==1.0)
        high_fidelity_obj = train_obj[np.where(train_x_full[:, -1]==1.0)].detach().numpy()
        accum_hf = [np.max(high_fidelity_obj[:i+1]) for i in range(len(high_fidelity_obj))]
        # Low-fidelity values
        lf_indices = np.where(train_x_full[:, -1] == 0.1)[0]
        low_fidelity_obj = train_obj[lf_indices].detach().numpy()
        accum_lf = [np.max(low_fidelity_obj[:i+1]) for i in range(len(low_fidelity_obj))]
        
        if len(hf_indices) > 0:
            plt.plot(cumulative_cost_array[hf_indices], accum_hf, label=f"{search_alg} (HF)")


        if len(lf_indices) > 0:
            plt.plot(cumulative_cost_array[lf_indices], accum_lf, label=f"{search_alg} (LF)", linestyle="--")

        
        #plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
        plt.xlabel('Accumulated Cost')
        plt.ylabel('Max High-Fidelity Evaluation')
        #plt.title(title)
        plt.legend(loc = 'lower right')
        plt.minorticks_on()
        plt.grid(True)
        plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   
        plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
        plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
        
        plt.show(block=False)

def plot_histogram(domain):
    high_fidelity = domain[np.where(domain[:, -2]==1.0)]
    low_fidelity = domain[np.where(domain[:, -2]!= 1.0)]
    correlation = np.corrcoef(high_fidelity[:, -1], low_fidelity[:, -1])[0,1]

    plt.hist(low_fidelity[:, -1], label=f'Low-fidelity Data (Correlation: {correlation:.2f})', bins=20, alpha=0.5, color='blue')
    plt.hist(high_fidelity[:, -1], label='High-fidelity Data', bins=20, alpha=0.5, color='red')
    # plt.title('Distribution of Evaluations')
    plt.xlabel('f(x)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.minorticks_on()
    plt.grid(True)
    plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   
    plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.show(block=False)


# This is code to debug the batch process a few cells later, showing the different runs individually, rather than averaged.
def plot_cost_batch_individual_old(domain, dictionary): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        for j in range(len(train_x_full)):
            cumulative_cost_array = np.array(cumulative_cost[j])
            #print('cumulative_cost_array',cumulative_cost_array)
            hf_indices = np.where(train_x_full[j][:, -1]==1.0)
            high_fidelity_obj = train_obj[j][np.where(train_x_full[j][:, -1]==1.0)].detach().numpy()
            accum_target = []
            for i in range(len(high_fidelity_obj)):
                accum_target.append(max(high_fidelity_obj[0:i+1]))

            plt.plot(cumulative_cost_array[hf_indices ], accum_target, label=f"{search_alg}_{j}")    
    #plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Target')
    plt.legend()
    plt.show(block=False)

import matplotlib.pyplot as plt
import numpy as np

def plot_cost_batch_individual(domain, dictionary): 
    # Compute global max reached
    max_yield_reached = max(
        np.max(train_obj[j][np.where(train_x_full[j][:, -1] == 1.0)].detach().numpy())
        for _, (train_x_full, train_obj, _) in dictionary.items()
        for j in range(len(train_x_full))
    )

    plt.figure(figsize=(10, 6))
    color_map = plt.get_cmap("tab10")  # Up to 10 distinct colors

    for idx, search_alg in enumerate(dictionary):
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        color = color_map(idx)  # Use the same color for all runs of this search_alg
        for j in range(len(train_x_full)):
            cumulative_cost_array = np.array(cumulative_cost[j])
            hf_indices = np.where(train_x_full[j][:, -1]==1.0)[0]
            high_fidelity_obj = train_obj[j][hf_indices].detach().numpy()

            accum_target = np.maximum.accumulate(high_fidelity_obj)
            plt.plot(cumulative_cost_array[hf_indices], accum_target, color=color, alpha=0.7)

        # Add a single legend entry for this algorithm
        plt.plot([], [], color=color, label=search_alg)

    plt.axhline(y=max_yield_reached, color='black', linestyle='--', label='Max Yield Reached')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Target')
    plt.title("Max Yield vs. Accumulated Cost")
    plt.legend(title="Search Algorithm", loc = 'lower right')
    plt.minorticks_on()
    plt.grid(True)
    plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   
    plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.show()


def plot_cost_batch_average(domain, dictionary, num_points=100):
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])
    plt.figure(figsize=(10, 6))
    
    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        
        # Define common cost grid for interpolation
        all_costs = []
        for j in range(len(train_x_full)):
            hf_indices = np.where(train_x_full[j][:, -1]==1.0)
            hf_costs = np.array(cumulative_cost[j])[hf_indices]
            all_costs.extend(hf_costs)
        min_cost = np.min(all_costs)
        max_cost = np.max(all_costs)
        common_cost_grid = np.linspace(min_cost, max_cost, num_points)

        interpolated_runs = []

        for j in range(len(train_x_full)):
            cumulative_cost_array = np.array(cumulative_cost[j])
            #print('cumulative_cost_array',cumulative_cost_array)
            hf_indices = np.where(train_x_full[j][:, -1]==1.0)[0]

            print(f"{search_alg} run {j}")
            print(f"  train_x_full shape: {train_x_full[j].shape}")
            print(f"  cumulative_cost_array shape: {cumulative_cost_array.shape}")
            print(f"  hf_indices: {hf_indices}")

            high_fidelity_obj = train_obj[j][hf_indices].detach().numpy()
            
            # Compute cumulative max
            accum_target = np.maximum.accumulate(high_fidelity_obj)
            hf_costs = cumulative_cost_array[hf_indices]

            print(f"  len(hf_costs): {len(hf_costs)}")
            print(f"  len(accum_target): {len(accum_target)}")

            # Interpolate onto common cost grid
            interp = interp1d(hf_costs, accum_target, kind='previous',
                              bounds_error=False, fill_value=(accum_target[0], accum_target[-1]))
            interpolated = interp(common_cost_grid)
            interpolated_runs.append(interpolated)
        
        interpolated_runs = np.array(interpolated_runs)
        mean_curve = np.mean(interpolated_runs, axis=0)
        std_curve = np.std(interpolated_runs, axis=0)

        # Plot mean and std deviation shading
        plt.plot(common_cost_grid, mean_curve, label=f"{search_alg} Mean")
        plt.fill_between(common_cost_grid, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Target')
    plt.legend(loc = 'lower right')
    plt.title('Averaged Max Yield vs Accumulated Cost')
    plt.minorticks_on()
    plt.grid(True)
    plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   
    plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.show()

def plot_cost_batch_average_simple(domain, dictionary, cost_step=0.1, max_budget=40.0): #change depending on lf_cost
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])
    plt.figure(figsize=(10, 6))

    cost_grid = np.arange(cost_step, max_budget + cost_step, cost_step)

    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        all_runs_yields = []

        for j in range(len(train_x_full)):
            # Get high-fidelity indices and corresponding costs
            hf_indices = np.where(train_x_full[j][:, -1] == 1.0)[0]
            hf_costs = np.array(cumulative_cost[j])[hf_indices]
            hf_yields = train_obj[j][hf_indices].detach().numpy()
            cummax_yields = np.maximum.accumulate(hf_yields)

            # Create an array to hold yield values at each point on cost grid
            yield_at_grid = np.zeros_like(cost_grid)
            current_yield = 0.0
            k = 0  # Index for hf_costs
            for i, cost in enumerate(cost_grid):
                if k < len(hf_costs) and cost >= hf_costs[k]:
                    current_yield = cummax_yields[k]
                    k += 1
                yield_at_grid[i] = current_yield

            all_runs_yields.append(yield_at_grid)

        all_runs_yields = np.array(all_runs_yields)
        mean_yield = np.mean(all_runs_yields, axis=0)
        std_yield = np.std(all_runs_yields, axis=0)

        plt.plot(cost_grid, mean_yield, label=f"{search_alg}")
        plt.fill_between(cost_grid, mean_yield - std_yield, mean_yield + std_yield, alpha=0.2)

    #plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Target')
    plt.legend(loc = 'lower right')
    plt.title('Averaged Max Yield vs Accumulated Cost')
    plt.minorticks_on()
    plt.grid(True)
    plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   
    plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
    plt.show()



'''
# Here we plot the mean maximum high-fidelity target reached so far with SD for a batch of experiments. 
def plot_cost_batch(domain, dictionary, title, allocated_budget, lf = 0.01, colour=['blue', 'red', 'green', 'yellow', 'orange']): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for id, search_alg in enumerate(dictionary):
        aggregate_max_target = []
        train_x_full_batch, train_obj_batch, cumulative_cost_batch = dictionary[search_alg]
        
        #We mltiply by 100 to get past the floating poitn arithmetic errors when dealing with fidelities and intervals of 0.1, say.
        cost_range = list(np.arange(1* 100,allocated_budget*100, int(lf*100)))
        
        for batch_no in range(len(train_x_full_batch)):
            #cumulative_cost_array = np.array([round(x,1) for x in cumulative_cost_batch[batch_no]])
            #hf_indices = np.where(train_x_full_batch[batch_no][:, -1]==1.0)
            #high_fidelity_obj = train_obj_batch[batch_no][np.where(train_x_full_batch[batch_no][:, -1]==1.0)].detach().numpy().squeeze(-1)
            #max_target = discretise_cost_and_maximise(high_fidelity_obj, cost_range, cumulative_cost_array, hf_indices)
            
            print('cumulative_cost_batch',cumulative_cost_batch)
            
            x = train_x_full_batch[batch_no]
            y = train_obj_batch[batch_no].detach().numpy().squeeze(-1)
            cost = np.array([round(c, 2) for c in cumulative_cost_batch[batch_no]])

            # Filter only HF points
            hf_mask = x[:, -1] == 1.0
            hf_costs = cost[hf_mask]
            hf_targets = y[hf_mask]

            #aggregate_max_target.append(max_target)
            max_target = discretise_cost_and_maximise(hf_targets, cost_range, hf_costs)
            aggregate_max_target.append(max_target)
        maximum_aggregate_mean = np.mean(aggregate_max_target, axis = 0)
        
        elts_above = []
        elts_below = []
        aggregate_max_target_swapped = np.swapaxes(aggregate_max_target,0,1)
        for idx, budget_elts in enumerate(aggregate_max_target_swapped):
            above_batch = []
            below_batch= []
            for budget_elt in budget_elts:
                if budget_elt >= maximum_aggregate_mean[idx]:
                    above_batch.append(budget_elt)
                if budget_elt <= maximum_aggregate_mean[idx]:
                    below_batch.append(budget_elt)
            elts_above.append(above_batch)
            elts_below.append(below_batch)
        maximum_aggregate_lowerbound = np.array([np.min(x) for x in elts_below])
        maximum_aggregate_upperbound =  np.array([np.max(x) for x in elts_above])

        cost_range_scaled = [x/100 for x in cost_range]
        plt.plot(cost_range_scaled, maximum_aggregate_mean, label=f'Mean {search_alg}', color=colour[id])
        plt.fill_between(x=cost_range_scaled, y1=maximum_aggregate_lowerbound, y2=maximum_aggregate_upperbound, color=colour[id], alpha=0.2, label=f'[Min({search_alg}), Max({search_alg})]')    
    plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Evaluation')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show(block=False)
'''

def plot_cost_batch(domain, dictionary, title, allocated_budget, lf = 0.1, colour=['blue', 'red', 'green', 'yellow', 'orange']): #change depending on lf_cost
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for id, search_alg in enumerate(dictionary):
        aggregate_max_target = []
        train_x_full_batch, train_obj_batch, cumulative_cost_batch = dictionary[search_alg]

        # We multiply by 100 to get past floating point arithmetic errors.
        cost_range = list(np.arange(1 * 100, allocated_budget * 100, int(lf * 100)))

        for batch_no in range(len(train_x_full_batch)):
            x = train_x_full_batch[batch_no]
            y = train_obj_batch[batch_no].detach().numpy().squeeze(-1)

            # Ensure cumulative_cost_batch is a list of floats for each batch
            # Check if cumulative_cost_batch[batch_no] is a list or a float
            cumulative_cost_array = cumulative_cost_batch[batch_no]
            
            # In case the cumulative_cost_batch has floats, directly handle them
            if isinstance(cumulative_cost_array, float):
                cumulative_cost_array = [cumulative_cost_array]  # Convert to list if it's a float

            cumulative_cost_array = np.array([round(c, 2) for c in cumulative_cost_array])

            # Filter only high-fidelity points
            hf_mask = x[:, -1] == 1.0
            hf_costs = cumulative_cost_array[hf_mask]
            hf_targets = y[hf_mask]

            # Call discretise_cost_and_maximise with hf_targets and hf_costs
            max_target = discretise_cost_and_maximise(hf_targets, cost_range, hf_costs)
            aggregate_max_target.append(max_target)

        # Compute the mean of all the max targets
        maximum_aggregate_mean = np.mean(aggregate_max_target, axis=0)

        # Compute the lower and upper bounds
        elts_above = []
        elts_below = []
        aggregate_max_target_swapped = np.swapaxes(aggregate_max_target, 0, 1)
        for idx, budget_elts in enumerate(aggregate_max_target_swapped):
            above_batch = []
            below_batch = []
            for budget_elt in budget_elts:
                if budget_elt >= maximum_aggregate_mean[idx]:
                    above_batch.append(budget_elt)
                if budget_elt <= maximum_aggregate_mean[idx]:
                    below_batch.append(budget_elt)
            elts_above.append(above_batch)
            elts_below.append(below_batch)
        
        # Compute the bounds
        maximum_aggregate_lowerbound = np.array([np.min(x) for x in elts_below])
        maximum_aggregate_upperbound = np.array([np.max(x) for x in elts_above])

        # Plotting
        cost_range_scaled = [x / 100 for x in cost_range]
        plt.plot(cost_range_scaled, maximum_aggregate_mean, label=f'Mean {search_alg}', color=colour[id])
        plt.fill_between(x=cost_range_scaled, y1=maximum_aggregate_lowerbound, y2=maximum_aggregate_upperbound, color=colour[id], alpha=0.2, label=f'[Min({search_alg}), Max({search_alg})]')

    plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Evaluation')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show(block=False)



def discretise_cost_and_maximise(high_fidleity_points, cost_range_input, cumulative_cost_array_input, hf_costs):
    accum_target = []
    # This part generates the accumulated maximum of just each high-fidleity point so far.
    for i in range(len(high_fidleity_points)):
        accum_target.append(max(high_fidleity_points[0:i+1]))

    accum_target = np.array(accum_target)
    max_target = []
    #We mltiply by 100 to get past the floating poitn arithmetic errors when dealing with fidelities and intervals of 0.1, say.
    #cumulative_cost_array_input_times_100 = np.array([int(x*100) for x in cumulative_cost_array_input])
    #for id, x in enumerate(cost_range_input):
        #if x in cumulative_cost_array_input_times_100[ hf_indices_input ]:
            #max_target.append(accum_target[np.where(cumulative_cost_array_input_times_100[ hf_indices_input ] == x )[0]][0].item())
        #elif id == 0:
            #max_target.append(0)
        #else:
            #max_target.append(max_target[id-1])
    #return max_target
    cumulative_cost_times_100 = np.array([int(x * 100) for x in hf_costs])
    for id, x in enumerate(cost_range_input):
        if x in cumulative_cost_times_100:
            max_target.append(accum_target[np.where(cumulative_cost_times_100 == x)[0]][0].item())
        elif id == 0:
            max_target.append(0)
        else:
            max_target.append(max_target[id-1])


ALLOCATED_BUDGET_BATCH = 50
# plot_cost_batch(domain, searchDictBatch, f'Domain {int(len(domain)/2)}: Average of {N_EXP_BATCH} Runs for {ALLOCATED_BUDGET_BATCH} Budget', ALLOCATED_BUDGET_BATCH, lf=0.1 )
def plot_pairwise(domain, searchDictBatch, main_key, lf, allocated_budget, no_of_expts ):
    keys_without = [x for x in searchDictBatch.keys()]
    keys_without.remove(main_key)
    no_of_rows = len(keys_without)
    figure=plt.figure(figsize=(10,20))
    for id, key in enumerate(keys_without):
        new_dict = {main_key: searchDictBatch[main_key], key: searchDictBatch[key]}
        plt.subplot(no_of_rows,1, id+1)
        plot_cost_batch(domain, 
                        new_dict,
                        title='', 
                        allocated_budget=ALLOCATED_BUDGET_BATCH, 
                        lf=lf)
    plt.suptitle(f'Domain {int(len(domain)/2)}: Average of {no_of_expts} Runs for {allocated_budget} Budget', y=0.9)
    save_image(figure)
    plt.show(block=False)