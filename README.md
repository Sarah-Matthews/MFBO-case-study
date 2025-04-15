# MFBO-case-study

## MFBO_py Contents


File contents falls into the categories below:

### 1. Set-up files

- **acqfs.py**  
  File containing the multi-fidelity MES and single-fidelity EI and MES acquisition functions. Also contains code to run optimisation cycle & batch processes
- **setup_file.py**
  File to setup benchmarking procedure including generation of initial sample, function to evaluate the emulator at different fidelity levels, and general functions e.g. compute_correlation to calculate the LF/HF data correlation.

### 2. Running benchmarking 

- **running.py**  
  Central file to run a benchmark procedure (single experiment or batch) for the different models (MF-MES, SF-MES, SF-EI).
  The functions recieve an allocated budget, initial sample size, low fidelity data cost, and variance of noise to be added in the creation of low-fidelity data.
  The initial search space and results dictionary are saved to files.


### 3. Results

- **SampleSpaces**  
  Folder contains CSV files of the initial sample space from which the initial sample is extracted.
- **SearchDictionaries**  
Folder contains text files with the exported search dictionaries from each model within the benchmarking process.
Dictionaries include a full set of x values (conditions) with associated objective values, fidelity level and cumulative cost count.


### 4. Plotting & analysis

- **plotting.py**  
  File contains the functions used for plotting relevant graphs (e.g. cumulative cost against maximum high-fidelity objective value)
- **plots.py**  
  File calls the graph functions defined in plotting.py for the input search dictionaries
