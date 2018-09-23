# Design philosophy

The idea is to design loosely connected modules. The package consists
of different modules that can be used independently up to a point.

For installing the package, please install the requirement file

`conda install --file requirements.txt`
or
`pip install -r requirements.txt`

## Overview
The main division can be found as follows:
- Modeling
- Methods
- Plotting
Please note that a unittest is written for nearly every different main file
# Modeling
Models form a general model that can be found in models.py. Through inheritance
one can define their own modes. Note that only a minimal set of properties are required
for the different models

## Modules:
- models.py : contains the main model mold [inherit your model from this main mold]
- fastIsing.py: numpy orientated Ising model
- XOR.py: contains XOR and AND functions for testing simple properties of methods
-simulate.py: currently only contains simulate for simulation, the idea is to expand this
or rework present functions for better cognitive separation. Ideas for the future are to check
whether cythonizing this piece of code yields faster code.

# Methods
This module contains the methods for the different information theoretical measures.
Currently, IDT for source and sinks are implemented.

## Modules:
- information.py: IDT and sampling methods (MCMC) and other miscellaneous functions

# plotting
This part of the model is responsible for plotting various parts of the results.


## Modules
- plotting.py contains functions to show graph (including adjacency) next to fits for
computing the IDT as well as computing the impact of the nudge on node distribution using
the Hellinger distance.


# TODO:
- General: clean up code; remove redundant information.
- move fastIsing in the models.py for cleaner outlook
