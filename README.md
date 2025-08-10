# dynamic-thermalization
This is the research code by Nikolay Gnezdilov, which tests thermalization in a few-qubit system, calculating exact dynamics. The thermal observables/fluctuations emerge at the averaging of a dynamical quantum mechanical observable over randomized variants of its unitary evolution under a Hamiltonian that falls into a class of Gaussian unitary ensemble of random matrix theory. The energy of the initial state determines the resulting temperature.
This code was used for the exact dynamics. The results are published in Communications Physics 8 (1), 95 (2025) by H. Perrin, T. Scoquart, A. I.Pavlov, and N. V. Gnezdilov. The experimental data (raw and processed) from the quantum computer is available on the Zenodo repository: https://zenodo.org/records/14639657.

Here is a brief description of the current repository:

SYK2_thermalizer.py -- .py file that contains the necessary functions to compute the exact dynamics in the considered few-qubit system.

Ising_thermalizer -- .py file that runs the thermalization protocol with a target system being a 1D Ising model with longitudinal and transverse fields.

‎couplings_QC_N=4_nr=100_J=1.npy -- the data for 100 realizations of random coupling constants sampled from the complex Gaussian distribution (with zero mean and finite variance) used in the experimental run.

‎run_quench_protocol.ipynb -- Jupyter notebook that runs the evolution.

‎levels_stats.ipynb -- Jupyter notebook that analyzes the level statistics of the total Hamiltonian of the system, which equals the target Hamiltonian plus a randomized all-to-all auxiliary interaction.

plotting.ipynb -- Jupyter notebook for plotting the data.
