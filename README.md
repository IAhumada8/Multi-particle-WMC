# About the WMC MPI C++ code - Two Particle Quantum Systems

Parallel implementation of the Yloop algorithm for Worldline Monte Carlo simulations of two-particle quantum systems using MPI.

## Overview

This code implements the Yloop algorithm in the Worldline Monte Carlo (WMC) formalism for simulating two-particle quantum systems with parallel computation using MPI. The algorithm generates Brownian loops to compute quantum propagators and extract ground state energies.

## Features

- **MPI Parallelization**: Distributed computation across multiple processes
- **Yloop Algorithm**: Efficient generation of quantum fluctuations
- **Automatic Analysis**: Built-in energy fitting and error estimation (optional)
- **Progress Tracking**: Real-time progress monitoring

## Quick Start

### Prerequisites

- MPI implementation (OpenMPI, MPICH, etc.)
- C++11 compatible compiler
- Linux/Unix environment (for cluster execution)

### Compilation

```bash
mpic++ -O3 -std=c++11 -o yloop_simulator main.cpp
