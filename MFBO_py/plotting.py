from setup_file import *
from acqfs import *


#functions for plotting graphs

def plot_fidelities(samples, targets, cumulated_cost, title, total_domain, colours=['red', 'blue', 'green']):
    max_in_space= np.max(total_domain[np.where(total_domain[:, -2] == 1.0)][:, -1])
    max_reached = targets[np.where(samples[:, -1] == 1.0)].max()
    index_array = np.where(targets == max_reached)[0]
    #print(index_array)
    for index in index_array:
        if samples[index, -1] == 1.0:
            index_of_max = index 
    #print(index_of_max)
    #index_of_max = index_array[0].item()
    samples = samples.detach().numpy()
    targets = targets.detach().numpy()
    fidelities = list(dict.fromkeys(np.round(samples[:, -1], 3)))
    fidelities.sort(reverse=True)
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
    plt.axhline(y=max_in_space, color='black', linestyle='--', label='Domain Optimum')
    plt.axvline(x=cumulated_cost[index_of_max], color='black', linestyle=':')
    plt.axhline(y=max_reached, color='black', linestyle=':', label='Obtained Optimum')
    
    # plt.legend(loc="lower right")
    plt.xlabel("Accumulated Cost")
    plt.ylabel("Evaluation")
    #plt.ylim([min(targets)-1, max_in_space +1])
    plt.title(title)

def plot_fidelities_for_dict(dictionary, dictionary_domain, allocated_budget=50, number_init=5):
    figure = plt.figure(figsize=(13,11))
    no_of_rows = math.ceil(len(dictionary.keys())/2)
    # for id, key in enumerate(dictionary):
    #     plt.subplot(no_of_rows, 2, id + 1)
    #     plot_fidelities(dictionary[key][0], dictionary[key][1], dictionary[key][2], f'{key}', dictionary_domain)
    for id, key in enumerate(list(dictionary.keys())):
        plt.subplot(no_of_rows, 2, id + 1)
        plot_fidelities(dictionary[key][0], dictionary[key][1], dictionary[key][2], f'{key}', dictionary_domain)
    handles, labels = plt.gca().get_legend_handles_labels()
    #for key in list(dictionary.keys())[-1:]:
        #plt.subplot(2, 2, 4)
        #plot_fidelities(dictionary[key][0], dictionary[key][1], dictionary[key][2], f'{key}', dictionary_domain)
    # plt.suptitle(f'{number_init} Initial Samples, {allocated_budget} Budget, Domain-size {int(len(dictionary_domain)/2)}',  size=16)
    plt.subplots_adjust(top=0.93)
    plt.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor = (0, -0.01, 1, 1),
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
            accum_target.append(max(high_fidelity_obj[0:i+1]))

        plt.plot(cumulative_cost_array[hf_indices ], accum_target, label=search_alg)    
        plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
        plt.xlabel('Accumulated Cost')
        plt.ylabel('Max High-Fidelity Evaluation')
        plt.title(title)
        plt.legend()
        plt.show(block=False)


def plot_histogram(domain):
    high_fidelity = domain[np.where(domain[:, -2]==1.0)]
    low_fidelity = domain[np.where(domain[:, -2]!= 1.0)]
    correlation = np.corrcoef(high_fidelity[:, -1], low_fidelity[:, -1])[0,1]

    plt.hist(low_fidelity[:, -1], label=f'Low-fidelity Data (Correlation: {str(correlation)[:3]})', bins=20, alpha=0.5, color='blue')
    plt.hist(high_fidelity[:, -1], label='High-fidelity Data', bins=20, alpha=0.5, color='red')
    # plt.title('Distribution of Evaluations')
    plt.xlabel('f(x)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show(block=False)


# This is code to debug the batch process a few cells later, showing the different runs individually, rather than averaged.
def plot_cost_batch_individual(domain, dictionary): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for search_alg in dictionary:
        train_x_full, train_obj, cumulative_cost = dictionary[search_alg]
        for j in range(len(train_x_full)):
            cumulative_cost_array = np.array(cumulative_cost[j])
            hf_indices = np.where(train_x_full[j][:, -1]==1.0)
            high_fidelity_obj = train_obj[j][np.where(train_x_full[j][:, -1]==1.0)].detach().numpy()
            accum_target = []
            for i in range(len(high_fidelity_obj)):
                accum_target.append(max(high_fidelity_obj[0:i+1]))

            plt.plot(cumulative_cost_array[hf_indices ], accum_target, label=f"{search_alg}_{j}")    
    plt.axhline(y=max_in_space, color='black', linestyle='--', label='Global Max for High-Fidelity')
    plt.xlabel('Accumulated Cost')
    plt.ylabel('Max High-Fidelity Target')
    plt.legend()
    plt.show(block=False)

# Here we plot the mean maximum high-fidelity target reached so far with SD for a batch of experiments. 
def plot_cost_batch(domain, dictionary, title, allocated_budget, lf = 0.1, colour=['blue', 'red', 'green', 'yellow', 'orange']): 
    #max_in_space= np.max(domain[np.where(domain[:, -2] == 1.0)])
    max_in_space = np.max(domain[np.where(domain[:, -2] == 1.0), -1])

    for id, search_alg in enumerate(dictionary):
        aggregate_max_target = []
        train_x_full_batch, train_obj_batch, cumulative_cost_batch = dictionary[search_alg]
        
        #We mltiply by 100 to get past the floating poitn arithmetic errors when dealing with fidelities and intervals of 0.1, say.
        cost_range = list(np.arange(1* 100,allocated_budget*100, int(lf*100)))
        
        for batch_no in range(len(train_x_full_batch)):
            cumulative_cost_array = np.array([round(x,1) for x in cumulative_cost_batch[batch_no]])
            hf_indices = np.where(train_x_full_batch[batch_no][:, -1]==1.0)
            high_fidelity_obj = train_obj_batch[batch_no][np.where(train_x_full_batch[batch_no][:, -1]==1.0)].detach().numpy().squeeze(-1)
            max_target = discretise_cost_and_maximise(high_fidelity_obj, cost_range, cumulative_cost_array, hf_indices)
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

def discretise_cost_and_maximise(high_fidleity_points, cost_range_input, cumulative_cost_array_input, hf_indices_input):
    accum_target = []
    # This part generates the accumulated maximum of just each high-fidleity point so far.
    for i in range(len(high_fidleity_points)):
        accum_target.append(max(high_fidleity_points[0:i+1]))

    accum_target = np.array(accum_target)
    max_target = []
    #We mltiply by 100 to get past the floating poitn arithmetic errors when dealing with fidelities and intervals of 0.1, say.
    cumulative_cost_array_input_times_100 = np.array([int(x*100) for x in cumulative_cost_array_input])
    for id, x in enumerate(cost_range_input):
        if x in cumulative_cost_array_input_times_100[ hf_indices_input ]:
            max_target.append(accum_target[np.where(cumulative_cost_array_input_times_100[ hf_indices_input ] == x )[0]][0].item())
        elif id == 0:
            max_target.append(0)
        else:
            max_target.append(max_target[id-1])
    return max_target


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