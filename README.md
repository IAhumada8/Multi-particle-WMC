# About the WMC MPI C++ code - Multi-Particle Quantum Systems

Parallel implementation of the Yloop algorithm for Worldline Monte Carlo simulations of multi-particle non-relativistic quantum systems using MPI.

## Overview

This code implements the Yloop algorithm in the Worldline Monte Carlo (WMC) formalism for simulating two(or more)-particle non-relativistic quantum systems with parallel computation using MPI. The algorithm generates Brownian loops to compute quantum propagators and extract ground state energies.

## Features

- **MPI Parallelization**: Distributed computation across multiple processes
- **Yloop Algorithm**: Efficient generation of quantum fluctuations
- **Automatic Analysis**: Built-in energy fitting (optional) and error estimation 
- **Progress Tracking**: Real-time progress monitoring

## Quick Start

### Prerequisites

- C++ compatible compiler
- MPI implementation (OpenMPI, MPICH, etc.)
- GNU Scientific Library (GSL) for C++ (for non-linear fits, Optional)
- Linux/Unix environment (for cluster execution)

### Compilation

- For personal computers or workstations
```bash
mpic++ -O3 -o your_executable 2PYloopsMPI.cpp
```

- For personal computers or workstations with GSL (if using non-linear fit)
```bash
mpic++ -O3 -o your_executable 3PYloopsMPI.cpp -lgsl -lgslcblas -lm
```

- For HPC clusters (Recommended flags)
```bash
mpic++ -O3 -funroll-loops -march=native -o your_executable 2PYloopsMPI.cpp
````

- For HPC clusters with GSL (if using non-linear fit)
```bash
mpic++ -O3 -funroll-loops -march=native -o  $(pkg-config --cflags --libs gsl) -Wl,-rpath,$(pkg-config --variable=libdir gsl):$(pkg-config --variable=libdir openblas) your_executable 3PYloopsMPI.cpp
````
NOTE: The flags required to run the code with the GSL library may vary depending on each cluster, contact your HPC admin to find out which flags to use.

### Code execution

- For personal computers or workstations
```bash
mpirun --use-hwthread-cpus -np 8 ./your_executable
```

- For personal computers or workstations (modifying the output file name)
```bash
mpirun --use-hwthread-cpus -np 8 ./your_executable --output="Your_file_name.txt"
```

- For HPC clusters
```bash
mpirun --use-hwthread-cpus -np 64 ./your_executable
````
NOTE: On SLURM-managed clusters, you can also choose your file name instead of the default, but you won't be able to override the name if you use the same name, which will cause a failure and the code won't run. It's recommended to use the default name or always use different names for the --output flag.

